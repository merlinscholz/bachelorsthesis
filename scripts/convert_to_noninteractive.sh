#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

jupyter nbconvert --to script $DIR/../udc.ipynb --output run --output-dir .