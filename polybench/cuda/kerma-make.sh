#!/bin/bash

## SETTINGS ##
GPU_ARCH=sm_52
LOGFILE="kerma-make.log"
##############



print_help() 
{
  echo "Usage: ./kerma-make.sh [Option] ... "
  echo "Options: "
  echo "-h, --help    : Print this help message"
  echo "-ir           : Generate LLVM IR files"
  echo "-exe          : Compile a binary (a.out)"
  echo "-v, --verbose : Verbose"
}

log() {
  echo -e "==================================="
  echo -e "[$1] $2/$3"
  echo -e "-----------------------------------"
  echo -e ">>> ${@:4}"
  echo -e "==================================="
  echo -e "==================================="         
  ${@:4}
  echo -e "===================================\n"
}

VERBOSE=false
IR=false
EXE=false
HELP=false
CLEAN=false

for var in "$@"
do
  [[ $var == '-v' || $var == '--verbose' ]] && VERBOSE=true
  [[ $var == '-ir' ]] && IR=true
  [[ $var == '-exe' ]] && EXE=true
  [[ $var == '-h' || $var == '--help' ]] && HELP=true
  [[ $var == '-clean' ]] && CLEAN=true
  [[ $var == '-silent' ]] && {
    SILENT=true
    VERBOSE=false
  }
done

$HELP && { print_help; exit 1; }

$CLEAN && {
  for d in *
  do
    if [ -d $d ]; then
      cd $d
      rm -f *.ll
      cd ..
    fi
  done
  $VERBOSE && echo "[i] Polybench .ll files removed"
  exit
}

$IR || $EXE || { echo "Nothing to do. Try -h for available options. Exiting."; exit 1; }

set -a

$SILENT || echo -ne "Compiling PolyBench GPU v1.0 ..."

## Setup CUDA location
CU_HOME=/usr/local/cuda
[[ -z "${CUDA_HOME}" ]] || CU_HOME=$CUDA_HOME
CU_LIB=$CU_HOME/lib64
CU_INC=$CU_HOME/include

## Setup CLANG location
CC=clang++
[[ -z "$LLVM_HOME" ]] || CC=$LLVM_HOME/bin/clang++
CC_FLAGS="-O1 -x cuda -L$CUDA_LIB --cuda-gpu-arch=$GPU_ARCH"
CC_LIBS="-lcudart_static -ldl -lrt -lm -pthread"
CC_FLAGS_IR="-Xclang -disable-llvm-passes -g -S -emit-llvm"


if $VERBOSE; then
  echo -e "\n[i] Using CUDA... $CU_LIB"
  echo "[i] Using Clang... $CC "   
  echo -ne "[i] Creating "
  $IR && echo -ne "[llvm-ir]"
  $EXE && echo -ne "[binary]"
  echo
fi

echo -e "[kerma-make][polybench] $(date +%F---%H:%M:%S)\n\n" > $LOGFILE

start=`date +%s`

for d in *
do
  if [ -d $d ]; then
    
    $VERBOSE && echo -ne "   - $d/"
    
    cd $d

    rm -f *.ll
    
    # loop not really needed as there is only one .cu file per directory
    for f in *.cu; do
      $VERBOSE && echo -ne "$f ... "
      $EXE && { log 'EXE' $d $f $CC $CC_FLAGS $f $CC_LIBS; } >> ../$LOGFILE 2>&1;
      $IR && { log 'IR' $d $f $CC $CC_FLAGS $CC_FLAGS_IR $f $CC_LIBS; } >> ../$LOGFILE 2>&1;
      echo -e "\n" >> ../$LOGFILE
    done

    $VERBOSE && echo "ok"
    
    cd ..
  fi
done

end=`date +%s`

$VERBOSE && echo "[i] Finished after $((end - start)) second(s)"
$VERBOSE && echo "[i] Clang output written at $(realpath ../$LOGFILE) " || $SILENT || echo " ok"
