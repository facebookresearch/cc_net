# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import cc_net.text_normalizer as txt


def test_unicode_punct():
    weird = "，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％"
    replaced = ',.,""""""""""\'::?!();- - . ~\'...-<>[]%'
    assert txt.replace_unicode_punct(weird) == replaced

    assert txt.remove_unicode_punct(weird) == ""


def test_numbers():
    weird = "０２３４５６７８９ | 0123456789"
    normalized = "000000000 | 0000000000"
    assert txt.normalize(weird, numbers=True) == normalized
    assert txt.normalize(weird, numbers=False) == weird
