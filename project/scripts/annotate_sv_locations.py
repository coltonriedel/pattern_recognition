import csv
import math
import sys


def main():
    if len(sys.argv) != 4:
        print "Usage: python " + sys.argv[0] \
                + " ublox_log.csv sv_position_log.sp3 output.csv"
        exit(1)

    ubx_filename = sys.argv[1]
    pos_filename = sys.argv[2]
    out_filename = sys.argv[3]

    # open and read input files
    # convert timestamps to gps TOW
    # select applicable records in ublox log
    # correlate SV postions and timestamps
    #   calculate true range and range to estimated position
    # write entire record to output file
    # close files
