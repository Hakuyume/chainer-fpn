import argparse
import numpy as np
import pickle
import re


conv_name = {'w': 'W', 'b': 'b'}
conv_bn_name = {
    'w': 'conv/W',
    'b': 'conv/b',
    'bn_s': 'bn/gamma',
    'bn_b': 'bn/beta',
}
fc_name = {'w': 'W', 'b': 'b'}


def rename(name):
    return rename_fpn(name) or rename_rpn(name) or rename_head(name)


def rename_fpn(name):
    m = re.fullmatch(r'(?:res_)?conv1_([wb]|bn_[sb])', name)
    if m:
        return 'extractor/base/conv1/{}'.format(conv_bn_name[m.group(1)])

    m = re.fullmatch(r'res(\d)_0_branch1_([wb]|bn_[sb])', name)
    if m:
        return 'extractor/base/res{}/a/residual_conv/{}' \
            .format(m.group(1), conv_bn_name[m.group(2)])

    m = re.fullmatch(r'res(\d)_(\d+)_branch2([a-c])_([wb]|bn_[sb])', name)
    if m:
        return 'extractor/base/res{}/{}/conv{}/{}' \
            .format(m.group(1),
                    'a' if m.group(2) == '0' else 'b{}'.format(m.group(2)),
                    {'a': 1, 'b': 2, 'c': 3}[m.group(3)],
                    conv_bn_name[m.group(4)])

    m = re.fullmatch(r'fpn_inner_res(\d)_\d+_sum_(?:lateral_)?([wb])', name)
    if m:
        return 'extractor/inner/{}/{}' \
            .format(int(m.group(1)) - 2, conv_name[m.group(2)])

    m = re.fullmatch(r'fpn_res(\d)_\d+_sum_([wb])', name)
    if m:
        return 'extractor/outer/{}/{}' \
            .format(int(m.group(1)) - 2, conv_name[m.group(2)])


def rename_rpn(name):
    m = re.fullmatch(r'conv_rpn_fpn2_([wb])', name)
    if m:
        return 'rpn/conv/{}'.format(conv_name[m.group(1)])

    m = re.fullmatch(r'rpn_bbox_pred_fpn2_([wb])', name)
    if m:
        return 'rpn/loc/{}'.format(conv_name[m.group(1)])

    m = re.fullmatch(r'rpn_cls_logits_fpn2_([wb])', name)
    if m:
        return 'rpn/conf/{}'.format(conv_name[m.group(1)])


def rename_head(name):
    m = re.fullmatch(r'fc(\d)_([wb])', name)
    if m:
        return 'head/fc{}/{}'.format(int(m.group(1)) - 5, fc_name[m.group(2)])

    m = re.fullmatch(r'bbox_pred_([wb])', name)
    if m:
        return 'head/loc/{}'.format(fc_name[m.group(1)])

    m = re.fullmatch(r'cls_score_([wb])', name)
    if m:
        return 'head/conf/{}'.format(fc_name[m.group(1)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('caffe2model')
    parser.add_argument('output')
    args = parser.parse_args()

    with open(args.caffe2model, mode='rb') as f:
        model = pickle.load(f, encoding='bytes')
    blobs = model[b'blobs']

    model = {}
    for name, value in blobs.items():
        if isinstance(name, bytes):
            name = name.decode()
        new_name = rename(name)
        if new_name is None:
            continue

        if new_name == 'extractor/base/conv1/conv/W':
            value = value[:, ::-1]
            print(name, '->', new_name, '(BGR -> RGB)')
        elif re.fullmatch(r'.+/loc/[Wb]', new_name):
            value = value.reshape((-1, 4) + value.shape[1:])[:, [1, 0, 3, 2]] \
                .reshape(value.shape)
            print(name, '->', new_name, '(xywh -> yxhw)')
        else:
            print(name, '->', new_name)

        model[new_name] = value

        m = re.fullmatch('(.+/bn)/gamma', new_name)
        if m:
            print('(zeros)', '->', '{}/avg_mean'.format(m.group(1)))
            model['{}/avg_mean'.format(m.group(1))] = np.zeros_like(value)
            print('(ones)', '->', '{}/avg_var'.format(m.group(1)))
            model['{}/avg_var'.format(m.group(1))] = np.ones_like(value)
            print(0, '->', '{}/N'.format(m.group(1)))
            model['{}/N'.format(m.group(1))] = 0

    with open(args.output, mode='wb') as f:
        np.savez_compressed(f, **model)


if __name__ == '__main__':
    main()
