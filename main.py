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

def _(vec):
    return vec / np.max(np.abs(vec))


def image_to_src(image_arr=None, image_pil=None):
    if image_pil is None:
        image_pil = PIL.Image.fromarray(image_arr)

    f = cStringIO.StringIO()
    image_pil.save(f, 'PNG')
    f.seek(0)
    return 'data:image/png;base64,{:s}'.format(base64.b64encode(f.read()))


def image_to_output(image_rgb,
                    pad_height,
                    pad_width,
                    filter_size,
                    patch_size,
                    sigma_detection,
                    sigma_refinement,
                    num_refinements):

    output = {
        'src': image_to_src(image_arr=image_rgb),
    }

    half_pad_height = pad_height / 2
    half_pad_width = pad_width / 2
    half_patch_size = patch_size / 2

    image_gray = 100 - skimage.color.rgb2lab(image_rgb / 255.0)[:, :, 0]
    (height, width) = image_gray.shape

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
        half_pad_height:height,
        half_pad_width:width,
    ]
    vec_w = np.mean(image_patch, axis=0)
    gf_w = scipy.ndimage.filters.gaussian_filter1d(vec_w, sigma=sigma_detection, mode='constant')
    gf_max_w = scipy.ndimage.filters.maximum_filter1d(gf_w, size=filter_size, mode='constant')
    ws = np.where(gf_w == gf_max_w)[0]


    (fig, axes) = plot.subplots(1, 1, squeeze=False, figsize=(8, 6))
    ax = axes[0, 0]
    ax.plot(_(gf_w), color='b')


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
        patch['src'] = image_to_src(image_arr=image_patch)
        output['patches'].append(patch)

    return output


@app.route('/', methods=['GET', 'POST'])
def main():
    '''
    r = sess.get('http://railway.hinet.net/ImageOut.jsp')
    image_pil = PIL.Image.open(cStringIO.StringIO(r.content))
    image_rgb = np.array(image_pil)
    '''

    npy = np.load('/Users/harry/Dropbox/dev/boleto2/img/imgs.npy')[10:30]

    outputs = []
    for (num_image, image_rgb) in enumerate(npy):
        outputs.append(image_to_output(
            image_rgb,
            pad_height=30,
            pad_width=20,
            filter_size=15,
            patch_size=20,
            sigma_detection=3,
            sigma_refinement=6,
            num_refinements=1,
        ))

    return flask.render_template(
        'index.html',
        outputs=outputs,
    )
