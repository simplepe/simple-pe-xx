# Contributing to simple-pe

## Getting started
simple-pe  lives in a git repository which is hosted here:
https://git.ligo.org/stephen-fairhurst/simple-pe. If you haven't already, you should
[fork](https://docs.gitlab.com/ee/gitlab-basics/fork-project.html) this
repository and clone your fork to your local machine, 

```bash
$ git clone git@git.ligo.org:albert.einstein/simple-pe.git
```

replacing the SSH url to that of your fork. You can then install `simple-pe` by
running,

```bash
$ cd pesummary
$ python setup.py install
```

which will install `simple-pe`. 

## Update your fork
If you already have a fork of `simple-pe`, and are starting work on a new
project you can link your clone to the main (`lscsoft`) repository and pull in
changes that have been merged since the time you created your fork, or last
updated by following the instructions below:

1. Link your fork to the main repository:

```bash
$ cd pesummary
$ git remote add lscsoft https://git.ligo.org/lscsoft/simple-pe
```

2. Fetch new changes from the `lscsoft` repo,

```bash
$ git fetch lscsoft
```

3. Merge in the changes,

```bash
$ git merge lscsoft/master
```

## Reporting issues
All issues should be reported through the
[issue workflow](https://docs.gitlab.com/ee/user/project/issues/). When
reporting an issue, please include as much detail as possible to reproduce the
error, including information about your operating system and the version of
code. If possible, please include a brief, self-contained code example that
demonstrates the problem.

## Merge Requests
If you would like to make a change to the code then this should be done through
the [merge-request workflow](https://docs.gitlab.com/ee/user/project/merge_requests/).
We recommend that before starting a merge request, you should open an issue
laying out what you would like to do. This lets a conversation happen early in
case other contributors disagree with what you'd like to do or have ideas
that will help you do it.

Once you have made a merge request, expect a few comments and questions from
other contributors who also have suggestions. Once all discussions are resolved,
core developers will approve the merge request and then merge it into master.

All merge requests should aim to either add one feature, solve a bug, address 
some stylistic issues, or add to the documentation. If multiple changes are
lumped together, this makes it harder to review.

All merge requests should also be recorded in the CHANGELOG.md.
This just requires a short sentence describing describing the change that you
have made.

## Creating a new feature branch
All changes should be developed on a feature branch, in order to keep them
separate from other work, simplifying review and merge once the work is done.

To create a new feature branch:

```bash
$ git fetch lscsoft
$ git checkout -b my-new-feature lscsoft/master
```


## Unit tests
Aren't written yet


## Code style
Code should be written in the [PEP8](https://www.python.org/dev/peps/pep-0008/)
style. 

## Documentation
Documentation strings should be written in the
[NumpyDoc style](https://numpydoc.readthedocs.io/en/latest/).
