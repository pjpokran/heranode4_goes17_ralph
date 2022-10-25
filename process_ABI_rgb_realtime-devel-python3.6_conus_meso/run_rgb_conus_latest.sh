#!/bin/bash

export PATH="/home/poker/miniconda3/bin/:$PATH"

cd /home/poker/goes17_ralph/process_ABI_rgb_realtime-devel-python3.6_conus_meso

cp /home/ldm/data/grb/conus/03/latest.nc /dev/shm/latest_conus_03.nc
cmp /home/ldm/data/grb/conus/03/latest.nc /dev/shm/latest_conus_03.nc > /dev/null
CONDITION=$?
#echo $CONDITION

while :; do

  until [ $CONDITION -eq 1 ] ; do
#     echo same
     sleep 5
     cmp /home/ldm/data/grb/conus/03/latest.nc /dev/shm/latest_conus_03.nc > /dev/null
     CONDITION=$?
  done
#  echo different
  sleep 25
  cp /home/ldm/data/grb/conus/03/latest.nc /dev/shm/latest_conus_03.nc
  /home/poker/miniconda3/bin/python process_ABI_test_one_latest.py
  cmp /home/ldm/data/grb/conus/03/latest.nc /dev/shm/latest_conus_03.nc > /dev/null
  CONDITION=$?
#  echo repeat

done

