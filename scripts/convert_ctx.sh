#!/usr/bin/env bash

ctx2geotiff_pipeline(){
  mroctx2isis from=$1 to=/tmp/convert_ctx/$1.0.cub
  spiceinit from=/tmp/convert_ctx/$1.0.cub
  ctxcal from=/tmp/convert_ctx/$1.0.cub to=/tmp/convert_ctx/$1.1.cub
  rm /tmp/convert_ctx/$1.0.cub
  ctxevenodd from=/tmp/convert_ctx/$1.1.cub to=/tmp/convert_ctx/$1.2.cub
  rm /tmp/convert_ctx/$1.1.cub
  isis2std from=/tmp/convert_ctx/$1.2.cub to=$1
  rm /tmp/convert_ctx/$1.2.cub
}

export -f ctx2geotiff_pipeline

mkdir -p /tmp/ctx2geotiff

find . -type f -name '*.IMG' > /tmp/ctx2geotiff/filelist.txt

parallel ctx2geotiff_pipeline < /tmp/ctx2geotiff/filelist.txt

rm -rf /tmp/convert_ctx
