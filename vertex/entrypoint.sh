#!/bin/sh
set -e

TUNE_MODE_ARG=""
NO_CACHE_ARG=""
SMOKE_TEST_ARG=""
SKIP_TRAIN_ARG=""
for arg in "$@"
do
    case $arg in
        --tune_mode=*)
        TUNE_MODE_ARG="$arg"
        ;;
        --no-cache)
        NO_CACHE_ARG="--no-cache"
        ;;
        --smoke-test)
        SMOKE_TEST_ARG="--smoke-test"
        ;;
        --skip-train)
        SKIP_TRAIN_ARG="--skip-train"
        ;;
    esac
done

echo "============================================="
echo " STEP 1: RUNNING HYPERPARAMETER TUNING "
echo "============================================="
echo "Command: python tune_hyperparams.py ${TUNE_MODE_ARG} ${NO_CACHE_ARG} ${SMOKE_TEST_ARG}"

python tune_hyperparams.py ${TUNE_MODE_ARG} ${NO_CACHE_ARG} ${SMOKE_TEST_ARG}


if [ -z "$SKIP_TRAIN_ARG" ]; then
    echo "\n============================================="
    echo " STEP 2: RUNNING FINAL MODEL TRAINING "
    echo "============================================="
    echo "Command: python run.py ${NO_CACHE_ARG}"
    python run.py ${NO_CACHE_ARG}
else
    echo "\n--- Skipping final model training as requested by --skip-train flag. ---"
fi


echo "\n============================================="
echo " ALL JOBS COMPLETED SUCCESSFULLY "
echo "============================================="