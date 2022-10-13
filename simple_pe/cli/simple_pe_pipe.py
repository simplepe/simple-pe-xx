#! /usr/bin/env python

__authors__ = [
    "Charlie Hoy <charlie.hoy@ligo.org>",
]


def command_line():
    from simple_pe_analysis import command_line as _analysis_command_line
    from simple_pe_filter import command_line as _filter_command_line
    parser = ArgumentParser(parents=[_analysis_command_line(), _filter_command_line()])
    return parser


def main(args=None):
    parser = command_line()
    opts, _ = parser.parse_known_args(args=args)


if __name__ == "__main__":
    main()
