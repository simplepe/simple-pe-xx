# shell script to analyse GW150914

# first define the metric directions
if [ -z ${METRIC_DIRECTIONS} ]; then
    METRIC_DIRECTIONS="chirp_mass symmetric_mass_ratio chi_align"
fi
# then define the precession directions
if [ -z ${PREC_DIRECTIONS} ]; then
    PREC_DIRECTIONS="symmetric_mass_ratio chi_align chi_p"
fi
# then define which waveform you wish to use
if [ -z ${WVF} ]; then
    WVF="IMRPhenomXPHM"
fi
# then run `simple_pe_pipe`
echo "simple_pe_pipe --strain H1:GWOSC:GW150914 L1:GWOSC:GW150914 --approximant ${WVF} --f_low 20 --f_high 2048 --accounting_group UNKNOWN --accounting_group_user albert.einstein --outdir ./outdir --trigger_parameters trigger_parameters.json --asd H1:GW150914_H1_asd.txt L1:GW150914_L1_asd.txt --metric_directions ${METRIC_DIRECTIONS} --precession_directions ${PREC_DIRECTIONS}"
simple_pe_pipe --strain H1:GWOSC:GW150914 L1:GWOSC:GW150914 --approximant ${WVF} --f_low 20 --f_high 2048 --accounting_group UNKNOWN --accounting_group_user albert.einstein --outdir ./outdir --trigger_parameters trigger_parameters.json --asd H1:GW150914_H1_asd.txt L1:GW150914_L1_asd.txt --metric_directions ${METRIC_DIRECTIONS} --precession_directions ${PREC_DIRECTIONS}
# then run the output
bash ./outdir/submit/bash_simple_pe.sh
