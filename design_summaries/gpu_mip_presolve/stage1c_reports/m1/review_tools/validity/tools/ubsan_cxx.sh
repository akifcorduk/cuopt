#!/usr/bin/env bash
exec c++ -fsanitize=undefined -fno-sanitize-recover=undefined \
  -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG "$@"
