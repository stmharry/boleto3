import base64
import cStringIO
import flask
import flask_bootstrap
import matplotlib.pyplot as plot
import numpy as np
import PIL.Image
import requests
import scipy.ndimage
import skimage.color

app = flask.Flask(__name__)
flask_bootstrap.Bootstrap(app)
sess = requests.Session()

def _(vec, scale=1.0, ref_vec=None):
    return (0.9 * vec / np.max(np.abs(ref_vec or vec)) + 1.0) * scale / 2.0


def to_src(image_arr=None, image_fig=None):
    f = cStringIO.StringIO()
    if image_arr is not None:
        image_pil = PIL.Image.fromarray(image_arr)
        image_pil.save(f, 'png')
    elif image_fig is not None:
        image_fig.savefig(f, dpi=80, format='png')

    f.seek(0)
    return 'data:image/png;base64,{:s}'.format(base64.b64encode(f.read()))


def image_to_output(image_rgb,
                    pad_height=30,
                    pad_width=20,
                    filter_size=10,
                    patch_size=20,
                    sigma_detection=3,
                    sigma_refinement=6,
                    num_refinements=1):

    output = {
        'src': to_src(image_arr=image_rgb),
        'debug': [],
    }

    half_pad_height = pad_height / 2
    half_pad_width = pad_width / 2
    half_patch_size = patch_size / 2

    (height, width, channel) = image_rgb.shape
    image_lab = skimage.color.rgb2lab(image_rgb / 255.0)
    image_gray = 100 - image_lab[:, :, 0]

    image_mean = np.mean(image_gray)
    image_gray = image_gray - image_mean
    image_gray = np.pad(
        image_gray,
        pad_width=[
            (half_pad_height, half_pad_height),
            (half_pad_width, half_pad_width),
        ],
        mode='constant',
    )

    image_patch = image_gray[
        half_pad_height:height + half_pad_height,
        half_pad_width:width + half_pad_width,
    ]
    vec_w = np.mean(image_patch, axis=0)
    gf_w = scipy.ndimage.filters.gaussian_filter1d(vec_w, sigma=sigma_detection, mode='constant')
    gf_max_w = scipy.ndimage.filters.maximum_filter1d(gf_w, size=filter_size, mode='constant')
    ws = np.where(gf_w == gf_max_w)[0]

    (fig, axes) = plot.subplots(1, 1, squeeze=False, figsize=(10, 3))
    ax = axes[0, 0]
    ax.imshow(image_rgb)
    ax.plot(_(vec_w, height), color='b')
    ax.plot(_(gf_w, height), color='r')
    ax.plot(ws, _(gf_w, height)[ws], color='r', linestyle='none', marker='o')
    ax.set_xlim([0, width])
    ax.set_ylim([height, 0])
    output['debug'].append({'src': to_src(image_fig=fig)})

    # DEBUG
    (image_l, image_a, image_b) = np.split(image_lab, 3, axis=2)
    image_lab = np.concatenate([
        np.mean(image_l) - image_l,
        image_a - np.mean(image_a),
        image_b - np.mean(image_b),
    ], axis=2)

    image_lab = np.pad(
        image_lab,
        pad_width=[
            (half_pad_height, half_pad_height),
            (half_pad_width, half_pad_width),
            (0, 0),
        ],
        mode='constant',
    )
    image_patch = image_lab[
        half_pad_height:height + half_pad_height,
        half_pad_width:width + half_pad_width,
    ]
    vec_w = np.mean(image_patch, axis=0)
    gf2_l_w = scipy.ndimage.filters.gaussian_filter1d(vec_w[:, 0], sigma=3, axis=0, order=0, mode='constant')
    gf2_ab_w = scipy.ndimage.filters.gaussian_filter1d(vec_w[:, 1:3], sigma=6, axis=0, order=2, mode='constant')
    gf2_ab_w = np.sqrt(np.sum(np.square(gf2_ab_w), axis=1))
    gf2_w = gf2_l_w * gf2_ab_w
    gf2_max_w = scipy.ndimage.filters.maximum_filter1d(gf2_w, size=filter_size, mode='constant')
    ws = np.where(gf2_w == gf2_max_w)[0]

    (fig, axes) = plot.subplots(1, 1, squeeze=False, figsize=(10, 3))
    ax = axes[0, 0]
    # ax.plot(_(gf2_w), color='r')
    # ax.plot(ws, _(gf2_w)[ws], color='r', linestyle='none', marker='o')
    ax.plot(_(gf2_l_w), color='g')
    ax.plot(_(gf2_ab_w), color='b')
    ax.set_xlim([0, width])
    output['debug'].append({'src': to_src(image_fig=fig)})

    output['patches'] = []
    for w in ws:
        patch = {}

        image_patch = image_gray[
            half_pad_height:height,
            w:w + pad_width,
        ]
        vec_h = np.mean(image_patch, axis=1)
        gf_h = scipy.ndimage.filters.gaussian_filter1d(vec_h, sigma=sigma_detection, mode='constant')
        h = np.argmax(gf_h)

        patch['std'] = np.std(gf_h)

        for num_refinement in range(num_refinements):
            image_patch = image_gray[
                h:h + pad_height,
                w:w + pad_width,
            ]
            vec_w = np.mean(image_patch, axis=0)
            gf_w = scipy.ndimage.filters.gaussian_filter1d(vec_w, sigma=sigma_refinement, mode='constant')
            delta_w = np.argmax(gf_w)
            w = w - half_pad_width + delta_w

            image_patch = image_gray[
                h:h + pad_height,
                w:w + pad_width,
            ]
            vec_h = np.mean(image_patch, axis=1)
            gf_h = scipy.ndimage.filters.gaussian_filter1d(vec_h, sigma=sigma_refinement, mode='constant')
            delta_h = np.argmax(gf_h)
            h = h - half_pad_height + delta_h

        if np.logical_or.reduce((
            h < half_patch_size,
            height - half_patch_size <= h,
            w < half_patch_size,
            width - half_patch_size <= w,
        )):
            continue

        image_patch = image_gray[
            h + half_pad_height - half_patch_size:h + half_pad_height + half_patch_size,
            w + half_pad_width - half_patch_size:w + half_pad_width + half_patch_size,
        ]
        image_patch = np.uint8((image_patch + image_mean) / 100 * 255)

        patch['pos'] = (h, w)
        patch['src'] = to_src(image_arr=image_patch)
        output['patches'].append(patch)

    return output


@app.route('/', methods=['GET', 'POST'])
def main():
    '''
    r = sess.get('http://railway.hinet.net/ImageOut.jsp')
    image_pil = PIL.Image.open(cStringIO.StringIO(r.content))
    image_rgb = np.array(image_pil)

    '''

    image_rgb = np.load('imgs.npy')[13]
    output = image_to_output(image_rgb)
    outputs = [output]

    return flask.render_template(
        'index.html',
        outputs=outputs,
    )
