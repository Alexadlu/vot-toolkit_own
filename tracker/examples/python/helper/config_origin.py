# config_coord

POLICY = {
    # data path
    'vid2015':'/home/jaehyuk/dataset/ILSVRC2015',
    'pretrained_model':'/home/jaehyuk/code/experiment/VIDloader/checkpoint.ckpt-1',

    # data augmentation
    'kGeneratedExamplesPerImage': 4,
    'lamda_shift': 5.,
    'lamda_scale': 15.,
    'min_scale': -0.4,
    'max_scale': 0.4,

    # network initialization
    'WIDTH': 227,
    'HEIGHT': 227,
    'num': 1,
    'anchors': [4., 4.], # netout mean is about 0.5
    'side': 13,
    'channels': 3,

    # train policy( total 1k : 10 min in jaehyuk PC)
    'BATCH_SIZE': 8,
    'NUM_EPOCHS': 1600, # video epoch(4k), total image: 800k,  img/video = 200
    # 4k / 8 * 200 = 100k(800k img, 1 epoch) -> 16h

    # train_loss
    'object_scale': 5,
    'noobject_scale': 1,
    'coord_scale': 5,
    'thresh': .6,
    'thresh_IOU': .6,

    # train_optimizer
    'step_values': [50000],
    'learning_rates': [0.00001, 0.000001],
    'momentum': 0.9,
    'momentum2': 0.999,
    'decay': 0.0005,
}
