#! /usr/bin/env python
#| This file is a part of the sferes2 framework.
#| Copyright 2009, ISIR / Universite Pierre et Marie Curie (UPMC)
#| Main contributor(s): Jean-Baptiste Mouret, mouret@isir.fr
#|
#| This software is a computer program whose purpose is to facilitate
#| experiments in evolutionary computation and evolutionary robotics.
#|
#| This software is governed by the CeCILL license under French law
#| and abiding by the rules of distribution of free software.  You
#| can use, modify and/ or redistribute the software under the terms
#| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
#| following URL "http://www.cecill.info".
#|
#| As a counterpart to the access to the source code and rights to
#| copy, modify and redistribute granted by the license, users are
#| provided only with a limited warranty and the software's author,
#| the holder of the economic rights, and the successive licensors
#| have only limited liability.
#|
#| In this respect, the user's attention is drawn to the risks
#| associated with loading, using, modifying and/or developing or
#| reproducing the software by the user in light of its specific
#| status of free software, that may mean that it is complicated to
#| manipulate, and that also therefore means that it is reserved for
#| developers and experienced professionals having in-depth computer
#| knowledge. Users are therefore encouraged to load and test the
#| software's suitability as regards their requirements in conditions
#| enabling the security of their systems and/or data to be ensured
#| and, more generally, to use and operate it in the same conditions
#| as regards security.
#|
#| The fact that you are presently reading this means that you have
#| had knowledge of the CeCILL license and that you accept its terms.

import os

def build(bld):
    print ("Entering directory `" + os.getcwd() + "/modules/'")
    bld.program('cxx', 'test',
                source = 'test_nn.cpp',
                includes = '. ../../',
                use = '',
                uselib = 'EIGEN BOOST BOOST_GRAPH BOOST_UNIT_TEST_FRAMEWORK',
                target = 'test_nn')

    bld.program('cxx', 'test',
                source = 'test_dnn.cpp',
                includes = '. ../../',
                use = 'sferes2',
                uselib = 'EIGEN BOOST BOOST_GRAPH BOOST_UNIT_TEST_FRAMEWORK BOOST_SERIALIZATION',
                target = 'test_dnn')

    bld.program('cxx', 'test',
                source = 'test_mlp.cpp',
                includes = '. ../../',
                use = 'sferes2',
                uselib = 'EIGEN BOOST BOOST_GRAPH BOOST_UNIT_TEST_FRAMEWORK BOOST_SERIALIZATION',
                target = 'test_mlp')

    bld.program('cxx', 'test',
                source = 'test_hyper_nn.cpp',
                includes = '. ../../',
                use = 'sferes2',
                uselib = 'EIGEN BOOST B,OOST_GRAPH BOOST_UNIT_TEST_FRAMEWORK BOOST_SERIALIZATION',
                target = 'test_hyper_nn')

    bld.program('cxx', 'test',
                source = 'test_dnn_ff.cpp',
                includes = '. ../../',
                use = 'sferes2',
                uselib = 'EIGEN BOOST BOOST_GRAPH BOOST_UNIT_TEST_FRAMEWORK BOOST_SERIALIZATION',
                target = 'test_dnn_ff')

    bld.program('cxx', 'test',
                source = 'test_osc.cpp',
                includes = '. ../../',
                use = 'sferes2',
                uselib = 'EIGEN BOOST BOOST_GRAPH BOOST_UNIT_TEST_FRAMEWORK BOOST_SERIALIZATION',
                target = 'test_osc')


    bld.program('cxx', 'program',
                source = 'bench_nn.cpp',
                includes = '. ../../',
                use = 'sferes2',
                uselib = 'EIGEN BOOST_GRAPH BOOST',
                target = 'bench_nn')
