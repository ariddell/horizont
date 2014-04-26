MODULES := horizont/_utils horizont/_lda horizont/_random
PYX_FILES := $(addsuffix .pyx, $(MODULES))
SO_FILES := $(addsuffix .cpython33.so, $(MODULES))

default: test

$(SO_FILES): $(PYX_FILES)
	python setup.py build_ext --inplace

test: $(SO_FILES)
	nosetests -x -l DEBUG --nologcapture -s -w /tmp horizont.tests

test-utils: $(SO_FILES)
	nosetests -x -l DEBUG --nologcapture -s -w /tmp horizont.tests.test_utils

test-lda: $(SO_FILES)
	nosetests -x -l DEBUG --nologcapture -s -w /tmp horizont.tests.test_lda_strips

test-polya-gamma: $(SO_FILES)
	nosetests -l DEBUG -s -w /tmp horizont.tests.test_polya_gamma horizont.tests.test_logistic_normal


clean:
	rm -f horizont/*.so horizont/*.c horizont/*.cpp

.dummy: test 
