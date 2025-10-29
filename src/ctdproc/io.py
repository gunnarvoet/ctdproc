#!/usr/bin/env python
# coding: utf-8
import errno
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import xarray as xr
import xmltodict
from munch import Munch, munchify
from scipy.signal import savgol_filter

from .calcs import calc_allsal
from .helpers import datetime2mtlb, mtlb2datetime


def CTDx(filename):
    c = CTDHex(filename).to_xarray()
    add_default_proc_params(c)
    return c


def add_default_proc_params(ds):
    ds.attrs["verbose"] = 1
    ds.attrs["bounds_p"] = [0.0, 6200.0]
    ds.attrs["bounds_t"] = [-2.0, 40.0]
    ds.attrs["bounds_c"] = [2.0, 6.0]
    ds.attrs["bounds_s"] = [20, 38]
    ds.attrs["spike_thresh_t"] = 0.5
    ds.attrs["spike_thresh_s"] = 0.1
    ds.attrs["prod_c"] = 5.0e-7
    ds.attrs["prod_t"] = 1.0e-4
    ds.attrs["prod_s"] = 1.0e-8
    ds.attrs["prod_p"] = 1.0
    ds.attrs["diff_c"] = 1.0e-1
    ds.attrs["diff_t"] = 1.0e-1
    ds.attrs["diff_s"] = 1.0e-3
    ds.attrs["diff_p"] = 2.0
    ds.attrs["wthresh"] = 0.1
    ds.attrs["plot_spectra"] = 0
    ds.attrs["plot_path"] = ""


class CTDHex(object):
    """
    Converter for Seabird CTD data in hex format. Initialize with full path to hex file.
    xml config file needs to be located in the same directory.

    TODO:
      - Add oxygen hysteresis
      - Convert fluorometer voltage
    """

    def __init__(self, filename):
        self.filename = filename

        self._map_attrs = dict(
            lon=dict(
                name="",
                long_name="longitude",
                standard_name="longitude",
                units="degree_east",
            ),
            lat=dict(
                name="",
                long_name="latitude",
                standard_name="latitude",
                units="degree_north",
            ),
            t1=dict(
                name="TemperatureSensor1",
                long_name="temperature",
                standard_name="sea_water_temperature",
                units="degree_C",
            ),
            t2=dict(
                name="TemperatureSensor2",
                long_name="temperature",
                standard_name="sea_water_temperature",
                units="degree_C",
            ),
            c1=dict(
                name="ConductivitySensor1",
                long_name="conductivity",
                standard_name="sea_water_conductivity",
                units="mS cm-1",
            ),
            c2=dict(
                name="ConductivitySensor2",
                long_name="conductivity",
                standard_name="sea_water_conductivity",
                units="mS cm-1",
            ),
            p=dict(
                name="PressureSensor",
                long_name="pressure",
                standard_name="sea_water_pressure",
                units="dbar",
                positive="down",
            ),
            oxygen1=dict(
                name="OxygenSensor1",
                long_name="oxygen",
                standard_name="mass_concentration_of_oxygen_in_sea_water",
                units="ml/L",
            ),
            alt=dict(
                name="AltimeterSensor",
                long_name="height above bottom",
                standard_name="height_above_sea_floor",
                units="m",
            ),
            spar=dict(name="SPAR_Sensor", long_name="SPAR"),
            fl=dict(name="FluoroSeapointSensor", long_name="fluorescence"),
            par=dict(name="PAR_BiosphericalLicorChelseaSensor", long_name="PAR"),
            trans=dict(name="WET_LabsCStar", long_name="transmissivity"),
            modcount=dict(name="modcount"),
        )

        volt_vars = ["oxygen1", "alt", "spar", "fl", "par", "trans"]
        self._mapnames_volt = {v: self._map_attrs[v] for v in volt_vars}

        # extract all data and metadata and convert to physical units
        self._extract_physical_data()

    def _extract_physical_data(self):
        self.read_xml_config()
        self.parse_hex()
        self.physicalunits()

    def _detect_missing_words(self):
        """
        Determine location of data entries in hex file line.

        Sometimes there is no SPAR sensor, in this case one hex word may be
        missing. It lives on voltage channel 9. There may also be words missing
        if less sensors are present.

        Locating the right data entries also depends on the number of bytes
        written per scan. We read this information from the header.

        This is still a bit of work in progress.

        For details, see p.67/68 in manual-11pV2_018.pdf
        """
        # Read xml config file if this has not happened yet
        if not hasattr(self, "cfgp"):
            self.read_xml_config()
        # Count the number of sensors in the configuration
        n_index = self.cfgp.loc["@index"].values.shape[0]
        # No SPAR sensor
        if (
            "14" not in self.cfgp.loc["@index"].values
            and "SPAR_Sensor" not in self.cfgp.loc["@index"].keys()
        ):
            self._hexoffset = -6
        else:
            self._hexoffset = 0
        # Word 8 missing (there is probably a better way to implement this)
        if (
              "14" not in self.cfgp.loc["@index"].values
              and self._bytes_per_scan == 38
              ):
            self._hexoffset = -12
        # Less voltage words can lead to a shorter data stream, see manual p. 68
        if "14" in self.cfgp.loc["@index"].values and self._bytes_per_scan == 48:
            self._extra_hexoffset = 8
        elif "14" in self.cfgp.loc["@index"].values and self._bytes_per_scan == 44 and n_index == 11:
            self._extra_hexoffset = 0
        elif "14" in self.cfgp.loc["@index"].values and self._bytes_per_scan == 44 and n_index == 12:
            self._extra_hexoffset = 8
        elif "14" in self.cfgp.loc["@index"].values and self._bytes_per_scan == 45 and n_index == 10:
            self._extra_hexoffset = 0
        elif "14" not in self.cfgp.loc["@index"].values and self._bytes_per_scan == 45:
            self._extra_hexoffset = 8
        elif "14" not in self.cfgp.loc["@index"].values and self._bytes_per_scan == 38:
            self._extra_hexoffset = 8
        else:
            self._extra_hexoffset = 0

    def parse_hex(self):  # noqa: C901
        # Generate data structure for converted data: 5 freq, 8 voltage channels
        tmp = {}
        for i in range(5):
            tmp["f{}".format(i)] = []
        for i in range(8):
            tmp["v{}".format(i)] = []
        tmp["modcount"] = []
        tmp["time"] = []
        tmp["spar"] = []
        tmp["pst"] = []
        tmp["ctdstatus"] = []
        tmp["lon"] = []
        tmp["lat"] = []

        out = dict(header="")

        with open(self.filename, "rt") as fin:
            # read header
            for line in fin:
                if len(line) == 1:
                    pass
                elif line[0] == "*":
                    out["header"] += line
                    # if l[0:5] == "*    ":
                    # i = l.find("=")
                    if line[0:17] == "* Number of Bytes":
                        i = line.find("=")
                        self._bytes_per_scan = int(line[i + 2 :])
                    if line[0:25] == "* Number of Voltage Words":
                        i = line.find("=")
                        self._voltage_words = int(line[i + 2 :])
                    if line[0:34] == "* Append System Time to Every Scan":
                        self._hex_has_time = True
                    else:
                        self._hex_has_time = False
                    if line[0:15] == "* System UTC = ":
                        self.header_time_str = line[15:35]
                else:
                    break
            self._detect_missing_words()
            # read data
            for line in fin:
                # parse frequency channels
                for k in tmp.keys():
                    if "f" in k:
                        i = int(k[1])
                        tmp[k].append(self._hexword2freq(line[slice(i * 6, i * 6 + 6)]))
                # parse voltage channels
                for i in range(4):
                    v1, v2 = self._hexword2volt(
                        line[slice((i + 5) * 6, (i + 5) * 6 + 6)]
                    )
                    tmp["v{}".format(i * 2)].append(v1)
                    tmp["v{}".format(i * 2 + 1)].append(v2)
                # parse other channels
                # spar
                if self._hexoffset == 0:
                    tmp["spar"].append(self._hexword2spar(line[slice(57, 60)]))
                # NMEA data
                lat, lon = self._hexword2lonlat(
                    line[60 + self._hexoffset : 74 + self._hexoffset]
                )
                tmp["lon"].append(lon)
                tmp["lat"].append(lat)
                # pressure sensor temperature and ctd status
                pst, ctdstatus = self._hexword2pstat(
                    line[
                        slice(
                            74 + self._hexoffset + self._extra_hexoffset,
                            78 + self._hexoffset + self._extra_hexoffset,
                        )
                    ]
                )
                tmp["pst"].append(pst)
                tmp["ctdstatus"].append(ctdstatus)
                # modcount
                tmp["modcount"].append(
                    int(
                        line[
                            78
                            + self._hexoffset
                            + self._extra_hexoffset : 80
                            + self._hexoffset
                            + self._extra_hexoffset
                        ],
                        16,
                    )
                )
                # time
                if self._hex_has_time:
                    tmp["time"].append(
                        int(
                            line[
                                86
                                + self._hexoffset
                                + self._extra_hexoffset : 88
                                + self._hexoffset
                                + self._extra_hexoffset
                            ]
                            + line[
                                84
                                + self._hexoffset
                                + self._extra_hexoffset : 86
                                + self._hexoffset
                                + self._extra_hexoffset
                            ]
                            + line[
                                82
                                + self._hexoffset
                                + self._extra_hexoffset : 84
                                + self._hexoffset
                                + self._extra_hexoffset
                            ]
                            + line[
                                80
                                + self._hexoffset
                                + self._extra_hexoffset : 82
                                + self._hexoffset
                                + self._extra_hexoffset
                            ],
                            16,
                        )
                    )

            # generate output array
            # frequency variables are always there
            freqvars = ["t1", "c1", "p", "t2", "c2"]
            for i, k in enumerate(freqvars):
                out[k] = np.array(tmp["f{}".format(i)])
            # voltage variables. see which ones we have
            for k, v in self._mapnames_volt.items():
                if v["name"] in self.cfgp.keys():
                    channel = int(self.cfgp[v["name"]]["@index"])
                    vchannel = channel - 5
                    if vchannel < 8:
                        out[k] = np.array(tmp["v{}".format(vchannel)])
                    elif vchannel == 9 and k == "spar":
                        out["spar"] = np.array(tmp["spar"])
            out["lon"] = np.array(tmp["lon"])
            out["lat"] = np.array(tmp["lat"])
            out["pst"] = np.array(tmp["pst"])
            out["ctdstatus"] = tmp["ctdstatus"]
            out["modcount"] = np.array(tmp["modcount"])
            if self._hex_has_time:
                out["time"] = np.array(tmp["time"])
            else:
                out["time"] = self._generate_time_vector(len(out["t1"]))
            if "spar" in tmp:
                out["spar"] = np.array(tmp["spar"])
            self.dataraw = munchify(out)

    def _hexword2freq_old(self, hex_str):
        """
        Convert Seabird hex data to frequency
        each byte is given as two hex digits
        each SB freq word is 3 bytes
        calculates freq from 3 byte word

        Parameters
        ----------
        hex : str
            6 character long hex string

        Returns
        -------
        f : float
            frequency

        Notes
        -----
        This is an older version that is a bit slower than the new _hexword2freq.
        """
        f = (
            int(hex_str[:2], 16) * 256
            + int(hex_str[2:4], 16)
            + int(hex_str[4:], 16) / 256
        )
        return f

    def _hexword2freq(self, hex_str):
        """
        Convert Seabird hex data to frequency
        each byte is given as two hex digits
        each SB freq word is 3 bytes
        calculates freq from 3 byte word

        Parameters
        ----------
        hex : str
            6 character long hex string

        Returns
        -------
        f : float
            frequency

        Notes
        -----
        This is a new version that is a bit faster than _hexword2freq_old.
        See https://github.com/gunnarvoet/ctdproc/issues/1#issuecomment-2439574942
        for details.
        """
        return int(hex_str, 16) / 256

    def _hexword2volt_old(self, hex_str):
        """
        Convert Seabird hex data to voltage
        each byte is given as two hex digits
        each SB voltage is 1.5 words (8 MSB + 4 LSB)
        calculates 2 voltages from 3 byte word

        Parameters
        ----------
        hex_str : str
            6 character long hex str

        Returns
        -------
        v1, v2 : float
            voltages for 2 channels

        Notes
        -----
        This is an older version that is a bit slower than the new _hexword2volt.
        """
        byte1 = format(int(hex_str[0:2], 16), "08b")
        byte2 = format(int(hex_str[2:4], 16), "08b")
        byte3 = format(int(hex_str[4:6], 16), "08b")

        v1 = int(byte1 + byte2[:4], 2)
        v2 = int(byte2[4:] + byte3, 2)

        v1 = 5 * (1 - v1 / 4095)
        v2 = 5 * (1 - v2 / 4095)

        return v1, v2

    def _hexword2volt(self, hex_str):
        """
        Convert Seabird hex data to voltage
        each byte is given as two hex digits
        each SB voltage is 1.5 words (8 MSB + 4 LSB)
        calculates 2 voltages from 3 byte word

        Parameters
        ----------
        hex_str : str
            6 character long hex str

        Returns
        -------
        v1, v2 : float
            voltages for 2 channels

        Notes
        -----
        This is a new version that is a bit faster than _hexword2volt_old.
        See https://github.com/gunnarvoet/ctdproc/issues/1#issuecomment-2455907445
        for details.
        """
        e = int(hex_str, 16) ^ 0xffffff
        v1 = (e >> 12)/819
        v2 = (e & 0xfff)/819
        return v1, v2

    def _hexword2lonlat(self, hex_str):
        """
        Convert Seabird lon/lat data in hex format.
        Each byte is given as two hex digits.
        Last two characters contain pos/neg information.
        More information on p. 43 in manual-11pV2_018.pdf

        Parameters
        ----------
        hex_str : str
            7 byte / character long hex string

        Returns
        -------
        lon, lat : float
            longitude and latitude
        """
        b = format(int(hex_str[12:14], 16), "08b")
        # newpos = int(b[7])
        lonneg = int(b[1])
        latneg = int(b[0])
        lat = (
            (-1) ** latneg
            * (
                int(hex_str[:2], 16) * 65536
                + int(hex_str[2:4], 16) * 256
                + int(hex_str[4:6], 16)
            )
            / 5e4
        )
        lon = (
            (-1) ** lonneg
            * (
                int(hex_str[6:8], 16) * 65536
                + int(hex_str[8:10], 16) * 256
                + int(hex_str[10:12], 16)
            )
            / 5e4
        )
        return lat, lon

    def _hexword2spar(self, hex_str):
        """
        Convert Seabird spar (Photosynthetically Active Radiation) hex data
        each byte is given as two hex digits
        each SB voltage is 1.5 words (8 MSB + 4 LSB)
        calculates SPAR voltages from 1.5 byte half word

        Parameters
        ----------
        hex_str : str
            3 character long hex string

        Returns
        -------
        spar : float
            Photosynthetically Active Radiation
        """
        byte1 = format(int(hex_str[0], 16), "04b")
        byte2 = format(int(hex_str[1:3], 16), "08b")
        spar = int(byte1 + byte2, 2) / 819
        return spar

    def _hexword2pstat(self, hex_str):
        """
        Convert Seabird pressure sensor temperature and ctd status from hex data.
        Each byte is given as two hex digits.
        12 bit number from 0-4095 represents P sensor temperature .
        4 bit CTD status:
          bit 0 = pump status = 1/0 = on/off
          bit 1 = bottom contact = 1/0 = no contact/contact
          bit 2 = water sampler confirm = 1/0 = deck unit detetcts/does not signal
          bit 3 = CTD modem carrier detects/does not detect deck unit = 1/0

        Parameters
        ----------
        hex_str : str
            4 character long hex string

        Returns
        -------
        pst : int
            Pressure sensor temperature
        ctdstatus : binary
            CTD status (see above)
        """
        byte1 = format(int(hex_str[0:2], 16), "08b")
        byte2 = format(int(hex_str[2:4], 16), "08b")
        pst = int(byte1 + byte2[:4], 2)
        ctdstatus = byte2[4:]
        return pst, ctdstatus

    def _find_xmlconfig(self):
        """Generate path to xml config file for current hex file.
        Config file needs to be in the same directory as the hex file."""
        pp = Path(self.filename)
        name = pp.stem
        # try upper case filename
        xmlfile = name.upper() + ".XMLCON"
        p = pp.parent
        self.xmlfile = p.joinpath(xmlfile)
        # use os.listdir to find the actual case of the filename if the upper
        # case did not work.
        if self.xmlfile.name not in os.listdir(os.path.dirname(self.xmlfile)):
            xmlfile = name.lower() + ".XMLCON"
            self.xmlfile = p.joinpath(xmlfile)

    def read_xml_config(self):
        """Read xml config file."""
        self._find_xmlconfig()
        try:
            with open(self.xmlfile) as fd:
                tmp = xmltodict.parse(fd.read())
        except OSError as e:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), e.filename)
        sa = tmp["SBE_InstrumentConfiguration"]["Instrument"]["SensorArray"]["Sensor"]
        # parse only valid sensors
        cfg = {}
        ti = 0
        ci = 0
        oi = 0
        for si in sa:
            keys = list(si.keys())
            for k in keys:
                if "@" not in k and k != "NotInUse":
                    if k == "TemperatureSensor":
                        ti += 1
                        kstr = "{}{}".format(k, ti)
                    elif k == "ConductivitySensor":
                        ci += 1
                        kstr = "{}{}".format(k, ci)
                    elif k == "OxygenSensor":
                        oi += 1
                        kstr = "{}{}".format(k, oi)
                    else:
                        kstr = k
                    cfg[kstr] = si.copy()
                    cfg[kstr]["cal"] = munchify(cfg[kstr][k])
                    del cfg[kstr][k]
        self.cfgp = pd.DataFrame(cfg)
        self._xml_coeffs_to_float()

    def _xml_coeffs_to_float(self):
        # Convert calibration coefficients to floats.
        keep_strings = [
            "@SensorID",
            "SerialNumber",
            "CalibrationDate",
            "UseG_J",
        ]
        for k in self.cfgp.keys():
            for ki in self.cfgp[k].cal.keys():
                if isinstance(self.cfgp[k]["cal"][ki], str):
                    if ki not in keep_strings:
                        self.cfgp[k]["cal"][ki] = float(self.cfgp[k]["cal"][ki])
                elif isinstance(self.cfgp[k]["cal"][ki], list):
                    for i, li in enumerate(self.cfgp[k]["cal"][ki]):
                        for kli in li.keys():
                            self.cfgp[k]["cal"][ki][i][kli] = float(
                                self.cfgp[k]["cal"][ki][i][kli]
                            )
            # We can't have None values in the xarray.Dataset later on
            # or otherwise it won't properly write to netcdf. Therefore,
            # convert any None items to 'N/A'
            for ki, v in self.cfgp[k].cal.items():
                if v is None:
                    self.cfgp[k].cal[ki] = "N/A"

    def physicalunits(self):
        # pressure
        self._p_atm = 10.1353  # why not 10.1325 dbar?
        self.data = Munch()
        self.data.lon = self.dataraw.lon
        self.data.lat = self.dataraw.lat
        self.data.p = (
            self._freq2pressure(
                self.dataraw.p, self.dataraw.pst, self.cfgp.PressureSensor.cal
            )
            - self._p_atm
        )
        self.data.t1 = self._freq2temp(
            self.dataraw.t1, self.cfgp.TemperatureSensor1.cal
        )
        self.data.t2 = self._freq2temp(
            self.dataraw.t2, self.cfgp.TemperatureSensor2.cal
        )
        ccal1 = self.cfgp.ConductivitySensor1.cal.Coefficients[1]
        self.data.c1 = self._freq2cond(
            self.dataraw.c1, self.data.t1, self.data.p, ccal1
        )
        ccal2 = self.cfgp.ConductivitySensor2.cal.Coefficients[1]
        self.data.c2 = self._freq2cond(
            self.dataraw.c2, self.data.t2, self.data.p, ccal2
        )
        if hasattr(self.dataraw, "alt"):
            self.data.alt = self._volt2alt(
                self.dataraw.alt, self.cfgp.AltimeterSensor.cal
            )
        if hasattr(self.dataraw, "fl"):
            # couldn't find any good calibration coefficients in the con file I looked at.
            self.data.fl = self.dataraw.fl
        if hasattr(self.dataraw, "trans"):
            self.data.trans = self._volt2trans(
                self.dataraw.trans, self.cfgp[self._mapnames_volt["trans"]["name"]].cal
            )
        if hasattr(self.dataraw, "par"):
            self.data.par = self._volt2par(
                self.dataraw.par, self.cfgp[self._mapnames_volt["par"]["name"]].cal
            )
        if hasattr(self.data, 'c1') and hasattr(self.data, 't1') and hasattr(self.data, 'p'):
            if hasattr(self.data, 'lon') and hasattr(self.data, 'lat'):
                lon, lat = self.data.lon, self.data.lat
            else:
                print("Warning: No position data, using default lon=0, lat=0")
                lon, lat = 0.0, 0.0
            SA, SP = calc_allsal(self.data.c1, self.data.t1, self.data.p, lon, lat)
            self.data.sal = SP
            self.data.SA = SA
            # Ensure attributes dictionary exists
            if not hasattr(self, '_map_attrs'):
                self._map_attrs = {}
            self._map_attrs["sal"] = {
                "name": "sal",
                "long_name": "Practical Salinity",
                "units": "PSU",
            }
            self._map_attrs["SA"] = {
                "name": "SA",
                "long_name": "Absolute Salinity",
                "units": "g/kg",
            }
        if hasattr(self.dataraw, "oxygen1"):
            if hasattr(self.data, 'time'):
                time = self.data.time
            elif hasattr(self.data, 'scan'):
                time = self.data.scan
            else:
                time = np.arange(len(self.dataraw.oxygen1))
            self.data.oxygen1 = self._volt2oxygen(
                self.dataraw.oxygen1,
                self.data.t1,
                self.data.sal,
                self.data.p,
                time,
                self.cfgp[self._mapnames_volt["oxygen1"]["name"]].cal,
                min_pressure=1.0
            )
        self.data.modcount = self.dataraw.modcount
        self._check_modcount_errors(self.data.modcount)
        self.data.dtnum = self.sbetime_to_mattime(self.dataraw.time)
        self.data.time = self.mattime_to_datetime64(self.data.dtnum)

    def _calculate_oxsol(self, temp, salinity):
        """
        Calculate oxygen solubility using Garcia and Gordon (1992).
        Returns oxygen solubility in ml/L.
        """
        T_scaled = np.log((298.15 - temp) / (273.15 + temp))

        # Garcia and Gordon (1992) coefficients
        A0 = 2.00907
        A1 = 3.22014
        A2 = 4.05010
        A3 = 4.94457
        A4 = -0.256847
        A5 = 3.88767
        B0 = -0.00624523
        B1 = -0.00737614
        B2 = -0.0103410
        B3 = -0.00817083
        C0 = -0.000000488682

        oxsol = np.exp(
            A0 + A1*T_scaled + A2*T_scaled**2 + A3*T_scaled**3 +
            A4*T_scaled**4 + A5*T_scaled**5 +
            salinity * (B0 + B1*T_scaled + B2*T_scaled**2 + B3*T_scaled**3) +
            C0 * salinity**2
        )

        return oxsol

    def _calculate_tau(self, temp, pressure, cal):
        """Calculate sensor time constant tau(T,P)."""
        D0 = float(cal.D0)
        D1 = float(cal.D1)
        D2 = float(cal.D2)
        tau20 = float(cal.Tau20)

        tau = tau20 * D0 * np.exp(D1*pressure + D2*(temp - 20))
        return tau

    def _calculate_dvdt_window(self, volt, time_seconds, window_seconds=2.0):
        """
        Calculate dV/dt over a window as specified in SBE documentation.
        If there are fewer than 2 points in the window (2s by default), dV/dt is set to zero.
        """
        # Convert the desired window length (in seconds) to samples
        dt = np.mean(np.diff(time_seconds))
        window_length = int(window_seconds / dt)

        # window_length must be odd and >= polyorder + 2
        if window_length % 2 == 0:
            window_length += 1
        window_length = max(window_length, 5)

        # Compute the first derivative directly
        dVdt = savgol_filter(volt, window_length=window_length, polyorder=1, deriv=1, delta=dt)

        return dVdt

    def _freq2pressure(self, freq, tc, pcal):
        """Calculates pressure given frequency pressure temperature compensation
        and pressure calibration structure pcal"""
        psi2dbar = 0.689476

        Td = pcal.AD590M * tc + pcal.AD590B

        c = pcal.C1 + Td * (pcal.C2 + Td * pcal.C3)
        d = pcal.D1 + Td * pcal.D2
        t0 = pcal.T1 + Td * (pcal.T2 + Td * (pcal.T3 + Td * (pcal.T4 + Td * pcal.T5)))
        t0f = 1e-6 * t0 * freq
        fact = 1 - (t0f * t0f)
        pres = psi2dbar * (c * fact * (1 - d * fact))
        pres = pcal.Slope * pres + pcal.Offset
        return pres

    def _freq2temp(self, freq, tcal):
        """Calculate  temperature given frequency and
        temperature calibration structure tcal
        D. Rudnick 01/06/05"""

        logf0f = np.log(tcal.F0 / freq)
        temp = (
            1 / (tcal.G + logf0f * (tcal.H + logf0f * (tcal.I + logf0f * tcal.J)))
        ) - 273.15
        return temp

    def _freq2cond(self, freq, temp, pres, ccal):
        """Calculates conductivity given frequency, temperature,
        pressure and conductivity calibration structure ccal.
        D. Rudnick 01/06/05"""

        ff = freq / 1000
        cond = (ccal.G + ff * ff * (ccal.H + ff * (ccal.I + ff * ccal.J))) / (
            10 * (1 + ccal.CTcor * temp + ccal.CPcor * pres)
        )
        return cond

    def _volt2alt(self, volt, acal):
        """Calculate altimeter data from voltage."""
        alt = volt * acal.ScaleFactor + acal.Offset
        return alt

    def _volt2trans(self, volt, transcal):
        """Calculate transmissometer data from voltage."""
        trans = volt * transcal.M + transcal.B
        return trans

    def _volt2par(self, volt, cal):
        """Calculate PAR data from voltage."""
        par = (
            cal.Multiplier
            * (10**9 * 10 ** ((volt - cal.B) / cal.M))
            / cal.CalibrationConstant
        ) + cal.Offset
        return par

    def _volt2oxygen(self, volt, temp, salinity, pressure, time, ocal,
                     use_tau_correction=True, min_pressure=1.0):
        """
        Convert oxygen voltage to ml/L using SBE43 equation.

        Parameters
        ----------
        use_tau_correction : bool
            Apply tau*dV/dt correction. Default is True.
        min_pressure : float
            Minimum pressure (dbar) below which oxygen is set to NaN (default 1.0).
        """
        equation_index = int(ocal.Use2007Equation) if hasattr(ocal, 'Use2007Equation') else 1
        cal = ocal.CalibrationCoefficients[equation_index]

        Soc = float(cal.Soc)
        Voffset = float(cal.offset)
        A = float(cal.A)
        B = float(cal.B)
        C = float(cal.C)
        E = float(cal.E)

        # Convert time to numeric seconds
        if hasattr(time, 'dtype') and np.issubdtype(time.dtype, np.datetime64):
            time_seconds = (time - time[0]) / np.timedelta64(1, 's')
        elif hasattr(time, 'dtype') and np.issubdtype(time.dtype, np.timedelta64):
            time_seconds = time / np.timedelta64(1, 's')
        else:
            time_seconds = np.asarray(time, dtype=float)

        # Calculate tau*dV/dt correction if requested (for moorings)
        if use_tau_correction:
            dVdt = self._calculate_dvdt_window(volt, time_seconds, window_seconds=2.0)
            tau = self._calculate_tau(temp, pressure, cal)
            tau_correction = tau * dVdt
        else:
            print("  Skipping tau*dV/dt correction...")
            tau_correction = 0.0

        # Calculate oxygen solubility
        oxsol = self._calculate_oxsol(temp, salinity)

        # Convert temperature to Kelvin
        K = temp + 273.15

        # SBE43 equation
        oxygen = Soc * (volt + Voffset + tau_correction) * \
                 oxsol * \
                 (1.0 + A*temp + B*temp**2 + C*temp**3) * \
                 np.exp(E * pressure / K)

        # Quality control
        oxygen = np.where(pressure < min_pressure, np.nan, oxygen)
        oxygen = np.where(salinity < 1.0, np.nan, oxygen)
        oxygen = np.where(oxygen < 0, np.nan, oxygen)
        oxygen = np.where(oxygen > 15, np.nan, oxygen)

        return oxygen

    def _check_modcount_errors(self, modcount):
        """Check for modcount errors."""
        dmc = np.diff(modcount)
        mmc = np.mod(dmc, 256)
        fmc = np.squeeze(np.where(mmc - 1))
        if np.any(fmc):
            print("Warning: {} bad modcounts".format(len(dmc[mmc != 1])))
            print("Warning: {} missing scans".format(np.sum(mmc[fmc])))

    def sbetime_to_mattime(self, dt):
        """Convert SBE time format to matlab time format."""
        dtnum = dt / 24 / 3600 + 719529
        return dtnum

    def mattime_to_sbetime(self, dt):
        """Convert matlab time format to SBE time format."""
        dtnum = (dt - 719529) * 24 * 3600
        return dtnum

    def mattime_to_datetime64(self, dtnum):
        """Convert Matlab time format to numpy datetime64 time format."""
        dt64 = mtlb2datetime(dtnum)
        return dt64

    def _generate_time_vector(self, length):
        """generate time vector in SBE time format from a start time string"""
        pt = pd.to_datetime(self.header_time_str)
        # generate a 24Hz time vector
        pr = pd.date_range(
            pt,
            periods=length,
            freq="{}ns".format(np.int64(np.round(1 / 24 * 1e9))),
        )
        # t = pr.to_numpy()
        mattime = datetime2mtlb(pr.to_numpy())
        sbetime = self.mattime_to_sbetime(mattime)
        return sbetime

    def to_mat(self, matname):
        """Save data in Matlab format."""
        ctdout = self.data.copy()
        ctdout.pop("time")
        # ctdout.pop('matlabtime')
        # ctdout['den'] = ctdout['pden']
        # ctdout.pop('pden')
        # ctdout['time'] = ctd['matlabtime']
        sio.savemat(matname, ctdout, format="5")

    def to_xarray(self):
        """Convert data into xarray.Dataset.

        Returns
        -------
        ds : xarray.Dataset
            CTD time series in Dataset format.
        """
        dsout = self.data.copy()
        dsout.pop("time")
        dsout.pop("dtnum")
        datavars = dsout.keys()
        ds_data = {var: (["time"], self.data[var]) for var in datavars}

        ds = xr.Dataset(
            data_vars=dict(ds_data),
            coords={
                "time": (
                    ["time"],
                    self.data["time"],
                    {
                        "long_name": "time",
                        "standard_name": "time",
                    },
                )
            },
        )
        # set attributes
        for k in ds.data_vars:
            v = self._map_attrs[k]
            name = v.pop("name")
            ds[k].attrs = {**ds[k].attrs, **v}
            if name not in self.cfgp:
                continue
            ds[k].attrs["serial_number"] = self.cfgp[name].cal["SerialNumber"]
            ds[k].attrs["calibration_date"] = self.cfgp[name].cal["CalibrationDate"]

        return ds


def prof_to_mat(matname, datad, datau):
    out = dict(
        datad=datad,
        datau=datau,
    )
    # a few adjustments before saving as .mat file
    for k, v in out.items():
        # matlab time as new variable
        # out[k]['datenum'] = (['time'], datetime2mtlb(v.time.data))
        # drop datetime64
        out[k].drop("time")
        # make coordinates variables so they are saved
        out[k] = v.reset_coords()
    sio.savemat(matname, out, format="5")
