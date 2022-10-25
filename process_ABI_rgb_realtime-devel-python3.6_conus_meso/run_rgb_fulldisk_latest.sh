#!/bin/bash

export PATH="/home/poker/miniconda3/bin/:$PATH"

cd /home/poker/goes17_ralph/process_ABI_rgb_realtime-devel-python3.6_conus_meso

cp /home/ldm/data/grb-west/fulldisk/02/latest.nc /dev/shm/latest_fulldisk_02.nc
cmp /home/ldm/data/grb-west/fulldisk/02/latest.nc /dev/shm/latest_fulldisk_02.nc > /dev/null
CONDITION=$?
#echo $CONDITION

while :; do

  until [ $CONDITION -eq 1 ] ; do
#     echo same
     sleep 5
     cmp /home/ldm/data/grb-west/fulldisk/02/latest.nc /dev/shm/latest_fulldisk_02.nc > /dev/null
     CONDITION=$?
  done
#  echo different
  sleep 60
  cp /home/ldm/data/grb-west/fulldisk/02/latest.nc /dev/shm/latest_fulldisk_02.nc
  /home/poker/miniconda3/bin/python process_ABI_test_one_fulldisk_latest.py
  cmp /home/ldm/data/grb-west/fulldisk/02/latest.nc /dev/shm/latest_fulldisk_02.nc > /dev/null
  CONDITION=$?
#  echo repeat

done

