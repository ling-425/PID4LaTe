#Calculate statistical indicators
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import pandas as pd
import feather
import numpy as np
import tensorflow as tf
from keras import backend as K


def iter(num,epo,model):
    print("epoch:",epo)
    number = num
    epoch =epo


    # our
    if model == 'our':
        predict = pd.read_csv('improve/tmp/mendota/out/predict_model2_+_'+str(number)+'_'+str(epoch)+'.csv')

        # print(predict)
        pred = predict.values
        pred = pred[:,0:3185]
        # print(pred)
        pred = np.array(pred)
        print(pred.shape)
        # print(pred[0,12])



    test_data = feather.read_dataframe(os.path.join('improve/tmp/mendota/train/similar_980_1/inputs/labels_test'+str(number)+'.feather'))
    new_dates = np.load(os.path.join('improve/tmp/mendota/train/similar_980_1/inputs/dates.npy'), allow_pickle=True)#4-3
    # print(new_dates)
    n_depths=50
    t_steps=new_dates.shape[0]
    # print(t_steps)

    n_steps=353
    N_sec=19


    te_date = test_data.values[:,0]
    te_depth = test_data.values[:,1]
    te_temp = test_data.values[:,2]
    # print(te_date)
    # print(dd)

    m_te = np.zeros([n_depths,t_steps])
    obs_te = np.zeros([n_depths,t_steps])
    k=0
    #dd = 0
    for i in range(new_dates.shape[0]):
        # print(k)
        # print(te_date.shape)
        if k>=te_date.shape[0]:
            break
        # print(new_dates[i])
        # print(te_date[k])
        while new_dates[i]==te_date[k]:
            d = min(int(te_depth[k]/0.5),n_depths-1)
    #        if m_te[d,i]==1:
    #            print(d,te_depth[k])
            m_te[d,i]=1
            obs_te[d,i]=te_temp[k]
            # print(te_temp[k])
            k+=1
            if k>=te_date.shape[0]:
                break
            if te_date[k]== '2014-02-03':
                m_te[d, i] = 0
    # for i in range(0,obs_te.shape[1]):
    #     print(obs_te[0,i])
    #our
    if model == 'our' :
        y_test = obs_te[:,0:3185]
        m_test = m_te[:,0:3185]


    # print(y_test.shape)
    # print(m_test.shape)

    pred_s = tf.reshape(pred,[-1,1])
    y_s = tf.reshape(y_test,[-1,1])
    m_s = tf.reshape(m_test,[-1,1])

    r_cost = tf.sqrt(tf.reduce_sum(tf.square(tf.multiply((pred_s-y_s),m_s)))/tf.reduce_sum(m_s))
    r_cost_mse= tf.reduce_sum(tf.square(tf.multiply((pred_s-y_s),m_s)))/tf.reduce_sum(m_s)
    r_cost_mae = tf.reduce_sum(tf.abs(tf.multiply((pred_s - y_s), m_s))) / tf.reduce_sum(m_s)
    epsilon = 1e-10
    mape = tf.reduce_sum(tf.abs(tf.multiply(((pred_s - y_s) / (y_s + epsilon)), m_s))) / tf.reduce_sum(m_s)


    print("mape:")
    with tf.Session() as sess:
        print(sess.run(mape))
    print("test_rmse:")
    with tf.Session() as sess:
        print(sess.run(r_cost))
    print("test_mse:")
    with tf.Session() as sess:
        print(sess.run(r_cost_mse))
    print("test_mae:")
    with tf.Session() as sess:
        print(sess.run(r_cost_mae))





    #train loss
    train_data = feather.read_dataframe(os.path.join('improve/tmp/mendota/train/similar_980_1/inputs/labels_train'+str(number)+'.feather'))

    tr_date = train_data.values[:,0]
    tr_depth = train_data.values[:,1]
    tr_temp = train_data.values[:,2]
    # print(tr_date)

    t_steps = new_dates.shape[0]#3350
    # print(t_steps)
    m_tr = np.zeros([n_depths,t_steps])#50*3350
    obs_tr = np.zeros([n_depths,t_steps])
    k=0
    #dd = 0
    for i in range(new_dates.shape[0]):#3350
        if k>=tr_date.shape[0]:
            break
        # print(new_dates[i])
        # print(tr_date[k])
        while new_dates[i]==tr_date[k]:
            d = min(int(tr_depth[k]/0.5),n_depths-1)
            m_tr[d,i]=1
            obs_tr[d,i]=tr_temp[k]
            k+=1

            if k>=tr_date.shape[0]:
                break
            if tr_date[k]== '2014-02-03':
                m_tr[d, i] = 0


    #our
    if model == 'our':
        y_test = obs_tr[:,0:3185]
        m_test = m_tr[:,0:3185]

    # print(y_test.shape)
    # print(m_test.shape)
    # print(y_test[0,12])
    # print(m_test[0,12])

    # pred_s = tf.reshape(pred,[-1,1])
    y_s = tf.reshape(y_test,[-1,1])
    m_s = tf.reshape(m_test,[-1,1])

    # r_cost = tf.sqrt(tf.reduce_sum(tf.square(tf.multiply((pred_s-y_s),m_s)))/tf.reduce_sum(m_s))
    # r_cost = K.sqrt(K.sum(K.square((pred_s-y_s)*m_s))/K.sum(m_s))
    r_cost = tf.sqrt(tf.reduce_sum(tf.square(tf.multiply((pred_s - y_s), m_s))) / tf.reduce_sum(m_s))
    r_cost_mse = tf.reduce_sum(tf.square(tf.multiply((pred_s - y_s), m_s))) / tf.reduce_sum(m_s)
    r_cost_mae = tf.reduce_sum(tf.abs(tf.multiply((pred_s - y_s), m_s))) / tf.reduce_sum(m_s)


    mape = tf.reduce_sum(tf.abs(tf.multiply(((pred_s - y_s) / (y_s + epsilon)), m_s))) / tf.reduce_sum(m_s)
    print("trainmape:")
    with tf.Session() as sess:
        print(sess.run(mape))


    print("train_rmse:")
    with tf.Session() as sess:
        print(sess.run(r_cost))
    print("train_mse:")
    with tf.Session() as sess:
        print(sess.run(r_cost_mse))
    print("train_mae:")
    with tf.Session() as sess:
        print(sess.run(r_cost_mae))



for i in range(6,7):
    # if i == 3:
    #     iter(1)
    # else:
    #     iter(i)
    model = 'our'
    epo = i*100
    for j in range(5,6):
        print("itre:", j)
        iter(j,epo,model)