from html4vision import Col, imagetable
import glob
import natsort


sefa_interp = glob.glob('./sefa_directions32/results/*')
sefa_imgs = natsort.natsorted(sefa_interp)

mdd_interp = glob.glob('./8_4_4_2_2_directions32/results/*')
mdd_imgs = natsort.natsorted(mdd_interp)

# table description
cols = [
    Col('id0', 'Direction ID'),     # make a column of 1-based indices
    Col('img', 'Sefa', sefa_imgs),  # specify image content for column 2
    Col('img', 'MDD', mdd_imgs),    # specify image content for column 3
]

# html table generation
imagetable(
        cols,
        title='Discovering Interpretable Directions',
        imsize=(1024, 1024),
        imscale=1, preserve_aspect=True)

