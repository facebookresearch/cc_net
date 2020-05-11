# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import func_argparse

import cc_net.mine


def main():
    func_argparse.parse_and_call(cc_net.mine.get_main_parser())


if __name__ == "__main__":
    main()
