import datetime
import gzip

import matplotlib.pyplot as plt
import numpy as np


class UND:
    """read UND citation aircraft data"""

    def __init__(self, filename):
        self.filename = filename

    def read_file(self):
        try:
            # Open the input file for reading.
            # Use the gzip module to read in compressed files.
            if ".gz" in self.filename:
                self.fhand = gzip.open(self.filename, "r")
            else:
                self.fhand = open(self.filename, "r")
        except OSError:
            print(
                "ERROR (readfile.py): Cannot read ASCII file: "
                + self.filename
                + ", please make sure the file is properly formatted."
            )

    def header(self):
        '''define vars in header'''
        # Need skiprows to read the right section of the file.
        skiprows = int(self.fhand.readline().split()[0])
        self.NLHEAD = skiprows

        # move to beginning to read header
        self.fhand.seek(0)
        self.header = [self.fhand.readline() for lines in range(skiprows)]

    def parse_header(self):
        """Parse the header information.
        All new-line characters are stripped from strings."""

        self.FFI = int(self.header[0].split()[1])
        self.ONAME = self.header[1].rstrip()
        self.ORG = self.header[2].rstrip()
        self.SNAME = self.header[3].rstrip()
        self.MNAME = self.header[4].rstrip()
        self.IVOL = int(self.header[5].split()[0])
        self.VVOL = int(self.header[5].split()[1])
        self.DATE = self.header[6][:10]
        self.RDATE = self.header[6][11:].rstrip()
        self.DX = float(self.header[7])
        self.XNAME = self.header[8].rstrip()
        self.NV = int(self.header[9])
        self.VSCAL = self.header[10].split()
        self.VMISS = self.header[11].split()
        self.VNAME = [self.header[12 + i].rstrip() for i in range(self.NV)]
        self.NSCOML = int(self.header[self.NV + 12])
        self.NNCOML = int(self.header[self.NV + 13])
        self.DTYPE = self.header[self.NV + 14].rstrip()
        self.VFREQ = self.header[self.NV + 15].rstrip()

        self.VDESC = self.header[self.NV + 16].split()
        self.VUNITS = self.header[self.NV + 17].split()

        # Read data values from file.
        _data = np.loadtxt(self.fhand, dtype="float", skiprows=self.NLHEAD).T
        print(self.VDESC)

        # Store data in dictionary.
        # Use "object_name.data['Time']" syntax to access data.
        self.data = dict(zip(self.VDESC, _data))

    def close_file(self):
        """Close input file."""
        self.fhand.close()

    def find_times(self):
        '''Find starting and ending times of the file and calculate the total
        duration.'''
        _strt = self.data[self.VDESC[0]][0]
        _endt = self.data[self.VDESC[0]][-1]
        self.start_time = self.sfm2hms(_strt)
        self.end_time = self.sfm2hms(_endt)

        self.time_hms = []
        for time_s in self.data["Time"]:
            self.time_hms.append(self.sfm2hms(time_s))

    def sfm2hms(self, time_s):
        '''Convert seconds from midnight to HHMMSS'''
        return str(datetime.timedelta(seconds=round(time_s)))

    def collect_times(self):
        '''get an array of all times in file'''
        time_hms = []
        for time in self.data['Time']:
            time_hms.append(str(datetime.timedelta(seconds=round(time))))

        return time_hms
