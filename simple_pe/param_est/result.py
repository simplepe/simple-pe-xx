import numpy as np
from simple_pe.param_est import pe, metric
from simple_pe import io
from simple_pe.localization.event import Event
from pesummary.utils.samples_dict import SamplesDict
from pesummary.gw.file.formats.base_read import GWSingleAnalysisRead


class Result(GWSingleAnalysisRead):
    """
    """
    def __init__(self, f_low=None, psd=None, approximant=None, bayestar_localization=None, data_from_matched_filter={}):

        self.f_low = f_low
        self.psd = psd
        self.hm_psd = io.calculate_harmonic_mean_psd(self.psd)
        self.approximant = approximant
        self.bayestar_localization = bayestar_localization
        self._template_parameters = data_from_matched_filter.get(
            "template_parameters", None
        )
        self._snrs = data_from_matched_filter.get(
            "snrs", {}
        )
        #self._alpha_net = data_from_matched_filter.get(
        #    "alpha_net", None
        #)
        #self._f_net = data_from_matched_filter.get(
        #    "f_net", None
        #)
        #self._distance_face_on = data_from_matched_filter.get(
        #    "distance_face_on", None
        #)
        self._sigma = data_from_matched_filter.get(
            "sigma", None
        )
        self._response_sigma = data_from_matched_filter.get(
            "response_sigma",
            0.07 # obtained from generating samples on the sky and averaging
        )
        self._metric = None
        self.mcmc_samples = False
        self.samples = None
        self.parameters = None
        self.extra_kwargs = {"sampler": {}, "meta_data": {}}
        
    @property
    def metric(self):
        return self._metric
    
    @property
    def template_parameters(self):
        return self._template_parameters
    
    @property
    def snrs(self):
        return self._snrs

    @property
    def left_snr(self):
        return self._snrs.get("left", None)

    @property
    def right_snr(self):
        return self._snrs.get("right", None)
    
    @property
    def alpha_net(self):
        return self._alpha_net

    @property
    def f_net(self):
        return self._f_net

    @property
    def distance_face_on(self):
        return self._distance_face_on

    @property
    def sigma(self):
        return self._sigma

    @property
    def response_sigma(self):
        return self._response_sigma

    @property
    def samples_dict(self):
        data = super(Result, self).samples_dict
        return pe.SimplePESamples(data)
    
    def generate_metric(
        self, metric_directions, template_parameters=None, dominant_snr=None,
        tolerance=0.01, max_iter=10
    ):
        if template_parameters is not None:
            self._template_parameters = template_parameters
        if dominant_snr is not None:
            self._snrs["22"] = dominant_snr
        self._metric = metric.find_metric_and_eigendirections(
            self.template_parameters, metric_directions, self.snrs['22'],
            self.f_low, self.hm_psd,  self.approximant, tolerance, max_iter
        )
    
    def generate_samples_from_metric(self, *args, npts=int(1e5), **kwargs):
        if self.metric is None:
            self.generate_metric(*args, **kwargs)
        samples = self.metric.generate_samples(int(npts))
        self.samples = np.array(samples.samples).T
        self.parameters = samples.parameters
        return samples

    def generate_samples_from_sky(self, npts=int(1e5), bayestar_localization=None):
        from pesummary.core.reweight import rejection_sampling
        single_point = all(param in self.template_parameters for param in ["ra", "dec"])
        if bayestar_localization is not None and single_point:
            raise ValueError(
                "Please specify either 'bayestar_localization' or provide "
                "an estimate of 'ra' and 'dec' but not both"
            )
        if single_point:
            ra = np.ones(npts) * self.template_parameters["ra"][0]
            dec = np.ones(npts) * self.template_parameters["dec"][0]
        elif bayestar_localization:
            from ligo.skymap.io.fits import read_sky_map
            import astropy_healpix as ah
            import healpy as hp
            probs, _ = read_sky_map(bayestar_localization)
            npix = len(probs)
            nside = ah.npix_to_nside(npix)
            codec, ra = hp.pix2ang(nside, np.arange(npix))
            _cache = {"ra": [], "dec": []}
            while len(_cache["ra"]) < npts:
                pts = SamplesDict({'ra': ra, 'dec': np.pi/2 - codec})
                pts = rejection_sampling(pts, probs)
                _cache["ra"] += list(pts["ra"])
                _cache["dec"] += list(pts["dec"])
            pts = SamplesDict({'ra': _cache["ra"], 'dec': _cache["dec"]})
            pts = pts.downsample(npts)
            ra = pts['ra']
            dec = pts['dec']
        else:
            ev = Event.from_snrs(
                net, self.snrs["ifo_snr"], self.snrs["ifo_time"],
                self.template_parameters['chirp_mass']
            )
            ev.calculate_mirror()
            ev.localize_all()
            if ev.localized >= 3:
                # generate points from localization region to calculate F+ and Fx
                coh = ev.localization['coh']
                if ev.mirror and (ev.mirror_loc['coh'].snr > ev.localization['coh'].snr):
                    coh = ev.mirror_loc['coh']
                ra, dec = coh.generate_samples(npts=npts, sky_weight=True)
            elif ev.localized == 1:
                # source is only localized by the antenna pattern
                ra = np.random.uniform(0, 2 * np.pi, npts * 100)
                dec = np.pi / 2 - np.arccos(np.random.uniform(-1, 1, len(ra)))
                f = np.zeros_like(ra)
                det = ev.__getattribute__(ev.ifos[0])
                for i, (r, d) in enumerate(zip(ra, dec)):
                    fp, fc = det.antenna_pattern(r, np.pi / 2 - d, ev.psi, ev.gps)
                    f[i] = np.sqrt(fp ** 2 + fc ** 2)
                pts = SamplesDict({'ra': ra, 'dec': dec})
                pts = rejection_sampling(pts, f ** 3).downsample(npts)
                ra = pts['ra']
                dec = pts['dec']
            else:
                raise KeyError(
                    f"Unable to localize event from SNRs. This could be because "
                    f"you are considering a network with less than 3 detectors, or "
                    f"because the IFO SNRs are <{threshold}. The recovered IFO "
                    f"SNRs are {', '.join([ifo + ':' + str(abs(event_snr['ifo_snr'][ifo])) for ifo in ifos])}"
                )
        self.samples = np.vstack([self.samples.T, ra, dec]).T
        self.parameters = self.parameters + ["ra", "dec"]
    
    def generate_all_posterior_samples(self, function=None, **kwargs):
        samples = self.samples_dict
        samples.generate_all_posterior_samples(function=function, **kwargs)
        self.samples = np.array(samples.samples).T
        self.parameters = samples.parameters
        
    def generate_samples_from_aligned_spin_template_parameters(
        self, metric_directions, prec_interp_dirs, hm_interp_dirs,
        dist_interp_dirs, modes=['33'], alpha_net=None, interp_points=7,
        template_parameters=None, dominant_snr=None,
        reweight_to_isotropic_spin_prior=True
    ):
        import time
        t0 = time.time()
        if template_parameters is not None:
            self._template_parameters = template_parameters
        if dominant_snr is not None:
            self._snrs['22'] = dominant_snr
        if alpha_net is not None:
            self._alpha_net = alpha_net
        self.generate_samples_from_metric(
            metric_directions, self.template_parameters, self.snrs['22'],
        )
        self.generate_samples_from_sky(bayestar_localization=self.bayestar_localization)
        if reweight_to_isotropic_spin_prior:
            self.reweight_samples(
                pe.isotropic_spin_prior_weight, dx_directions=self.metric.dx_directions
            )
        self.generate_all_posterior_samples(
            function=pe.calculate_interpolated_snrs,
            psd=self.psd,
            f_low=self.f_low,
            dominant_snr=self.snrs['22'],
            modes=modes,
            #alpha_net=self.alpha_net,
            response_sigma=self.response_sigma,
            #fiducial_distance=self.distance_face_on,
            fiducial_sigma=self.sigma,
            dist_interp_dirs=dist_interp_dirs,
            hm_interp_dirs=hm_interp_dirs,
            prec_interp_dirs=prec_interp_dirs,
            interp_points=interp_points,
            approximant=self.approximant,
            left_snr=self.left_snr,
            right_snr=self.right_snr,
            template_parameters=self.template_parameters,
            snrs=self.snrs,
        )
        samples = self.samples_dict
        self.reweight_samples(
            pe.reweight_based_on_observed_snrs,
            hm_snr={'33': self.snrs['33']},
            prec_snr=self.snrs['prec'],
            snr_2pol={
                "not_right": samples['not_right'], "not_left": samples["not_left"]
            }, ignore_debug_params=['p_', 'weight']
        )
        print(f"Total time taken: {time.time() - t0:.2f}s")
        return self.samples_dict
