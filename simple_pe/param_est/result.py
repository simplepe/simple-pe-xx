import numpy as np
from simple_pe.param_est import pe, metric
from pesummary.utils.samples_dict import SamplesDict
from pesummary.gw.file.formats.base_read import GWSingleAnalysisRead


class Result(GWSingleAnalysisRead):
    """
    """
    def __init__(self, f_low=None, psd=None, approximant=None, data_from_matched_filter={}):

        self.f_low = f_low
        self.psd = psd
        self.approximant = approximant
        self._template_parameters = data_from_matched_filter.get(
            "template_parameters", None
        )
        self._snrs = data_from_matched_filter.get(
            "snrs", {}
        )
        self._alpha_net = data_from_matched_filter.get(
            "alpha_net", None
        )
        self._f_net = data_from_matched_filter.get(
            "f_net", None
        )
        self._distance_face_on = data_from_matched_filter.get(
            "distance_face_on", None
        )
        self._sigma = data_from_matched_filter.get(
            "sigma", None
        )
        self._response_sigma = data_from_matched_filter.get(
            "response_sigma", None
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
            self.f_low, self.psd,  self.approximant, tolerance=0.01, max_iter=10
        )
    
    def generate_samples_from_metric(self, *args, npts=1e6, **kwargs):
        if self.metric is None:
            self.generate_metric(*args, **kwargs)
        samples = self.metric.generate_samples(int(npts))
        self.samples = np.array(samples.samples).T
        self.parameters = samples.parameters
        return samples
    
    def generate_all_posterior_samples(self, function=None, **kwargs):
        samples = self.samples_dict
        samples.generate_all_posterior_samples(function=function, **kwargs)
        self.samples = np.array(samples.samples).T
        self.parameters = samples.parameters
        
    def generate_samples_from_aligned_spin_template_parameters(
        self, metric_directions, prec_interp_dirs, hm_interp_dirs,
        dist_interp_dirs, modes=['33'], alpha_net=None, interp_points=7,
        template_parameters=None, dominant_snr=None,
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
            metric_directions, self.template_parameters, self.snrs['22']
        )
        self.generate_all_posterior_samples(
            function=pe.calculate_interpolated_snrs,
            psd=self.psd,
            f_low=self.f_low,
            dominant_snr=self.snrs['22'],
            approximant=self.approximant,
            modes=modes,
            alpha_net=self.alpha_net,
            response_sigma=self.response_sigma
            fiducial_distance=self.distance_face_on,
            fiducial_sigma=self.sigma,
            dist_interp_dirs=dist_interp_dirs,
            hm_interp_dirs=hm_interp_dirs,
            prec_interp_dirs=prec_interp_dirs,
            interp_points=interp_points
        )
        self.reweight_samples(
            pe.reweight_based_on_observed_snrs,
            hm_snr={'33': self.snrs['33']},
            prec_snr=self.snrs['prec'],
            snr_2pol=self.snrs['not_right'],
            ignore_debug_params=['p_', 'weight']
        )
        print(f"Total time taken: {time.time() - t0:.2f}s")
        return self.samples_dict
