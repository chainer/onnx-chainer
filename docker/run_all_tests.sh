#!/bin/bash

bash docker/run_tests.sh 3.5 5.0.0
bash docker/run_tests.sh 3.6 5.0.0
bash docker/run_tests.sh 3.7 5.0.0

bash docker/run_tests.sh 3.5 6.0.0a1
bash docker/run_tests.sh 3.6 6.0.0a1
bash docker/run_tests.sh 3.7 6.0.0a1
