rm -rf new-project

mkdir new-project
cd new-project
asimov init "Simple PE Tests"

asimov apply -f https://git.ligo.org/asimov/data/-/raw/main/defaults/testing-pe.yaml
asimov apply -f https://git.ligo.org/asimov/data/-/raw/main/defaults/production-pe-priors.yaml

asimov apply -f https://git.ligo.org/asimov/data/-/raw/main/events/gwtc-2-1/GW150914_095045.yaml
asimov apply -f ../asimov-blueprint.yaml

#asimov manage build
#asimov manage submit
