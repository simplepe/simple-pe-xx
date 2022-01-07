    
#def sky_ring(object):
#    """
#    class to hold the details of a localization method
#    """
#
#    def __init__(self, event, npoints=500):
#        self.ifos = event.ifos
#        self.gps_time = date.XLALGreenwichMeanSiderealTimeToGPS(event.gmst)
#        self.npoints = npoints
#        ifo1 = getattr(event, ifos[0])
#        ifo2 = getattr(event, ifos[1])
#        self.dt = date.XLALArrivalTimeDiff(ifo1.location, ifo2.location, event.ra, event.dec, self.gps)
#        self.ring = sky.ISOTimeDelayLine([ifo1, ifo2], self.ra, self.dec, self.gps_time, n=self.npoints)
#        self.ra_ring = zeros(len(self.ring))
#        self.dec_ring = zeros(len(self.ring))
#        self.localization = {}
#        methods = ["coh", "left", "right", "marg"]
#        for method in methods:
#            self.localization[method] = []
#        self.f_ring = {}
#        for ifo in ifos:
#            self.f_ring[ifo] = zeros(len(ring), dtype=complex)
#        for i in xrange(len(ring)):
#            self.ra_ring[i] = self.ring[i].longitude
#            self.dec_ring[i] = self.ring[i].latitude
#            for ifo in self.ifos:
#                f_plus, f_cross = inject.XLALComputeDetAMResponse(getattr(event,ifo).response,
#                    self.ring[i].longitude, self.ring[i].latitude, 0, event.gmst)
#            self.f_ring[ifo][i] = complex(f_plus, f_cross)
#            #for method in methods:
#            #    info =
#
#frac = {}
#found = {}
#for method in ["sensitivity", "sutton"]:
#    frac[method] = zeros_like(probs)
#    found[method] = zeros_like(probs)
#    ring_vol = estimate_distance(ifos, ranges, f_ring, snr, method) ** 3
#    sig_vol = estimate_distance(ifos, ranges, f, snr, method) ** 3
#    s_vol = sort(ring_vol)[::-1]
#    for i, prob in enumerate(probs):
#        vol_n = argmax(cumsum(s_vol) > prob * sum(ring_vol))
#        vol_thresh = s_vol[vol_n]
#        frac[method][i] = 1. * vol_n / len(ring_vol)
#        found[method][i] = sig_vol > vol_thresh
#
## calculate left and right circular snrs:
#sky_tmplt = array([ranges[ifo] * f_ring[ifo] for ifo in ifos]).transpose()
#sig_tmplt = array([ranges[ifo] * f[ifo] for ifo in ifos])
#sig_prob = sum(sig_tmplt * sig_tmplt.conjugate()) ** (-1.5)
#denom = (sky_tmplt * sky_tmplt.conjugate()).sum(axis=1)
#sig_den = (sig_tmplt * sig_tmplt.conjugate()).sum()
#sig_snrsq = sum(sig_snr * sig_snr.conjugate())
#snrsq_l = abs((sig_snr * sky_tmplt.conjugate()).sum(axis=1)) ** 2 / denom
#snrsq_r = abs((sig_snr * sky_tmplt).sum(axis=1)) ** 2 / denom
#prob_l = exp((snrsq_l - sig_snrsq) / 2) / denom ** 1.5
#prob_r = exp((snrsq_l - sig_snrsq) / 2) / denom ** 1.5
#fprob = prob_l + prob_r
#s_prob = sort(fprob)[::-1]
#method = "face"
#frac[method] = zeros_like(probs)
#found[method] = zeros_like(probs)
#for i, prob in enumerate(probs):
#    vol_n = argmax(cumsum(s_vol) > prob * sum(ring_vol))
#    vol_thresh = s_vol[vol_n]
#    frac[method][i] = 1. * vol_n / len(ring_vol)
#    found[method][i] = sig_vol > vol_thresh
#    face_n = argmax(cumsum(s_prob) > prob * sum(s_prob))
#    face_thresh = s_prob[face_n]
#    frac["face"][i] = 1. * face_n / len(fprob)
#    found["face"][i] = sig_prob > face_thresh

