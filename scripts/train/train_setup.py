"""
Logic for model creation, training launching and actions needed to be
accomplished during training (metrics monitor, model saving etc.)
"""

import os
import time
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from protonet import TrainEngine
from protonet.models import Prototypical
from protonet.datasets import load

def train(config):
    np.random.seed(2019)
    tf.random.set_seed(2019)

    # Useful data
    model_type = config['model.type']
    now = datetime.now()
    now_as_str = now.strftime('%Y_%m_%d-%H:%M:%S')

    # Output files
    model_file = f"{config['model.save_path'].format(model_type, now_as_str)}"
    config_file = f"{config['output.config_path'].format(model_type, now_as_str)}"
    csv_output_file = f"{config['output.train_path'].format(model_type, now_as_str)}"
    train_summary_file = f"{config['summary.save_path'].format('train', model_type, now_as_str)}"
    test_summary_file = f"{config['summary.save_path'].format('test', model_type, now_as_str)}"
    csv_output_map_file = f"results/{config['data.dataset']}/protonet/{config['data.dataset']}_protonet_results.csv"
    summary_file = f"results/summary.csv"
    
    # Output dirs
    data_dir = f"data/{config['data.dataset']}"
    model_dir = model_file[:model_file.rfind('/')]
    config_dir = config_file[:config_file.rfind('/')]
    results_dir = csv_output_file[:csv_output_file.rfind('/')]

    # Create folder for model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Create folder for config
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    # Create output for train process
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # generate config file
    file = open(config_file, 'w')
    file.write(json.dumps(config, indent=2))
    file.close()

    file = open(f"{csv_output_file}", 'w') 
    file.write("epoch, loss, accuracy, val_loss, val_accuracy\n")
    file.close()

    train_summary_writer = tf.summary.create_file_writer(train_summary_file)
    val_summary_writer = tf.summary.create_file_writer(test_summary_file)

    # create summary file if not exists
    if not os.path.exists(summary_file):
        file = open(summary_file, 'w')
        file.write("datetime, model, config, min_loss, min_loss_accuracy\n")
        file.close()

    # create map file if not exists
    if not os.path.exists(csv_output_file):
        file = open(csv_output_file, 'w')
        file.write("datetime,config,trained_model,result,train_summary,test_summary\n")
        file.close()

    file = open(csv_output_map_file, 'a+') 
    file.write("{},{},{},{},{},{}\n".format(now_as_str,
                                            config_file,
                                            model_file,
                                            csv_output_file,
                                            train_summary_file,
                                            test_summary_file))
    file.close()

    # Data loader
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    ret = load(data_dir, config, ['train', 'val'])
    train_loader = ret['train']
    val_loader = ret['val']

    # Determine device
    if config['data.cuda']:
        cuda_num = config['data.gpu']
        device_name = f'GPU:{cuda_num}'
    else:
        device_name = 'CPU:0'

    # Setup training operations
    n_support = config['data.train_support']
    n_query = config['data.train_query']
    w, h, c = list(map(int, config['model.x_dim'].split(',')))
    model = Prototypical(n_support, n_query, w, h, c, nb_layers=config['model.nb_layers'], nb_filters=config['model.nb_filters'])
    optimizer = tf.keras.optimizers.Adam(config['train.lr'])

    # Metrics to gather
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    train_acc = tf.keras.metrics.Mean(name='train_accuracy')
    val_acc = tf.keras.metrics.Mean(name='val_accuracy')

    # Val losses for patience
    val_losses = []
    min_loss = [100]
    min_loss_acc = [0]

    @tf.function
    def loss(support, query):
        loss, acc = model(support, query)
        return loss, acc

    @tf.function
    def train_step(loss_func, support, query):
        # Forward & update gradients
        with tf.GradientTape() as tape:
            loss, acc = model(support, query)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))

        # Log loss and accuracy for step
        train_loss(loss)
        train_acc(acc)

    @tf.function
    def test_step(loss_func, support, query):
        loss, acc = loss_func(support, query)
        val_loss(loss)
        val_acc(acc)

    # Create empty training engine
    # FIXME: use keras model.fit
    train_engine = TrainEngine()

    # Set hooks on training engine
    def on_start(state):
        print("Training started.")
    train_engine.hooks['on_start'] = on_start

    def on_end(state):
        print("Training ended.")
    train_engine.hooks['on_end'] = on_end

    def on_start_epoch(state):
        print(f"Epoch {state['epoch']} started.")
        train_loss.reset_states()
        val_loss.reset_states()
        train_acc.reset_states()
        val_acc.reset_states()
    train_engine.hooks['on_start_epoch'] = on_start_epoch

    def on_end_epoch(state):
        print(f"Epoch {state['epoch']} ended.")
        epoch = state['epoch']
        template = 'Epoch {}, Loss: {}, Accuracy: {}, ' \
                   'Val Loss: {}, Val Accuracy: {}'

        file = open(csv_output_file, 'a+') 
        file.write("{}, {}, {}, {}, {}\n".format(epoch + 1,
                                                 train_loss.result(),
                                                 train_acc.result() * 100,
                                                 val_loss.result(),
                                                 val_acc.result() * 100)) 
        file.close()

        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_acc.result() * 100,
                              val_loss.result(),
                              val_acc.result() * 100))

        cur_loss = val_loss.result().numpy()
        if cur_loss < state['best_val_loss']:
            print("Saving new best model with loss: {}".format(cur_loss))
            state['best_val_loss'] = cur_loss
            min_loss[0] = cur_loss
            min_loss_acc[0] = val_acc.result()
            model.save(model_file)
        val_losses.append(cur_loss)

        # Early stopping
        patience = config['train.patience']
        if len(val_losses) > patience \
                and max(val_losses[-patience:]) == val_losses[-1]:
            state['early_stopping_triggered'] = True

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_acc.result(), step=epoch)
            train_loss.reset_states()           
            train_acc.reset_states()        

        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', val_acc.result(), step=epoch)
            val_loss.reset_states()          
            val_acc.reset_states()

    train_engine.hooks['on_end_epoch'] = on_end_epoch

    def on_start_episode(state):
        if state['total_episode'] % 20 == 0:
            print(f"Episode {state['total_episode']}")
        support, query = state['sample']
        loss_func = state['loss_func']
        train_step(loss_func, support, query)
    train_engine.hooks['on_start_episode'] = on_start_episode

    def on_end_episode(state):
        # Validation
        val_loader = state['val_loader']
        loss_func = state['loss_func']
        for i_episode in range(config['data.episodes']):
            support, query = val_loader.get_next_episode()
            test_step(loss_func, support, query)
    train_engine.hooks['on_end_episode'] = on_end_episode

    time_start = time.time()

    with tf.device(device_name):
        train_engine.train(
            loss_func=loss,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config['train.epochs'],
            n_episodes=config['data.episodes'])

    time_end = time.time()

    file = open(summary_file, 'a+') 
    summary = "{}, {}, protonet, {}, {}, {}\n".format(now_as_str,
                                                     config['data.dataset'],
                                                     config_file,
                                                     min_loss[0],
                                                     min_loss_acc[0])
    file.write(summary)

    elapsed = time_end - time_start
    h, min = elapsed//3600, elapsed%3600//60
    sec = elapsed-min*60

    print(f"Training took: {h} h {min} min {sec} sec")
