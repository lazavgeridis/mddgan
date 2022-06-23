
ATTRIBUTES = {
        # StyleGAN2
        'stylegan2_ffhq1024':
            {'initial_image':('all', 5.0),
             'long_chin':('3', 5.0),
             'open_mouth':('4-5', -5.0),
             'eyes_closed':('7', -5.0),
             #'hat_accessory':('0-2', -5.0),
             'eyeglasses':('0-2', -5.0),
             #'compress':('2-3', 5.0),
             'long_hair':('2-3', 5.0)
            },

        'stylegan2_car512':
            {'initial_image':('all', 5.0),
             'side':('0-4', 5.0),
             'sport':('4-5', -5.0),
             'colour':('10-11', 5.0)
            },

        # StyleGAN
        'stylegan_animeface512':
            {'initial_image':('all', 5.0),
            #'zoom':('0-1', 5.0),
            #'angry':('2-3', 5.0),
            #'surprise':('2-3', 5.0),
            #'eyes_closed':('2-3', -5.0),
             'open_mouth':('4-5', -5.0),
             'big_eyes':('4-5', 5.0),
             'style':('6-7', -5.0),
             'sketch':('8-15', 5.0),
             #'hair':('2-4', -5.0)
             'panel':('2-6', 5.0)
            },

        'stylegan_celebahq1024':
            {'age':('5-7', 0.0),
             'eyeglasses':('0-5', 0.0),
             'gender':('0-1', 0.0),
             'pose':('0-3', 0.0),
             'smile':('2-3', 0.0)
            }

        'stylegan_ffhq1024':
            {'age':('2,4,5,6', 0.0),
            'eyeglasses':('0-17', 0.0),
            'gender':('2-3', 0.0),
            'pose':('0-6', 0.0),
            'pose_inverted':('0-3', 0.0),
            'smile':('3', 0.0)
            }
}
