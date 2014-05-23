//| This file is a part of the sferes2 framework.
//| Copyright 2009, ISIR / Universite Pierre et Marie Curie (UPMC)
//| Main contributor(s): Jean-Baptiste Mouret, mouret@isir.fr
//|
//| This software is a computer program whose purpose is to facilitate
//| experiments in evolutionary computation and evolutionary robotics.
//| 
//| This software is governed by the CeCILL license under French law
//| and abiding by the rules of distribution of free software.  You
//| can use, modify and/ or redistribute the software under the terms
//| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
//| following URL "http://www.cecill.info".
//| 
//| As a counterpart to the access to the source code and rights to
//| copy, modify and redistribute granted by the license, users are
//| provided only with a limited warranty and the software's author,
//| the holder of the economic rights, and the successive licensors
//| have only limited liability.
//|
//| In this respect, the user's attention is drawn to the risks
//| associated with loading, using, modifying and/or developing or
//| reproducing the software by the user in light of its specific
//| status of free software, that may mean that it is complicated to
//| manipulate, and that also therefore means that it is reserved for
//| developers and experienced professionals having in-depth computer
//| knowledge. Users are therefore encouraged to load and test the
//| software's suitability as regards their requirements in conditions
//| enabling the security of their systems and/or data to be ensured
//| and, more generally, to use and operate it in the same conditions
//| as regards security.
//|
//| The fact that you are presently reading this means that you have
//| had knowledge of the CeCILL license and that you accept its terms.

#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE dnn_ff

#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/serialization/nvp.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <boost/graph/depth_first_search.hpp>

#include <sferes/fit/fitness.hpp>
#include <sferes/gen/evo_float.hpp>
#include <sferes/phen/parameters.hpp>
#include "gen_dnn_ff.hpp"
#include "phen_dnn.hpp"

using namespace sferes;
using namespace sferes::gen::dnn;
using namespace sferes::gen::evo_float;
using boost::archive::xml_oarchive;
using boost::archive::xml_iarchive;
using boost::serialization::make_nvp;

struct Params
{
  struct evo_float
  {
    SFERES_CONST float mutation_rate = 0.1f;
    SFERES_CONST float cross_rate = 0.1f;
    SFERES_CONST mutation_t mutation_type = polynomial;
    SFERES_CONST cross_over_t cross_over_type = sbx;
    SFERES_CONST float eta_m = 15.0f;
    SFERES_CONST float eta_c = 15.0f;
  };
  struct parameters
  {
    // maximum value of parameters
    SFERES_CONST float min = -5.0f;
    // minimum value
    SFERES_CONST float max = 5.0f;
  };
  struct dnn
  {
    SFERES_CONST size_t nb_inputs	= 4;
    SFERES_CONST size_t nb_outputs	= 1;
    SFERES_CONST size_t min_nb_neurons	= 4;
    SFERES_CONST size_t max_nb_neurons	= 5;
    SFERES_CONST size_t min_nb_conns	= 100;
    SFERES_CONST size_t max_nb_conns	= 101;

    SFERES_CONST float m_rate_add_conn	= 1.0f;
    SFERES_CONST float m_rate_del_conn	= 0.1f;
    SFERES_CONST float m_rate_change_conn = 1.0f;
    SFERES_CONST float m_rate_add_neuron  = 1.0f;
    SFERES_CONST float m_rate_del_neuron  = 1.0f;

    SFERES_CONST int io_param_evolving = true;
    SFERES_CONST init_t init = ff;
  };
};

BOOST_AUTO_TEST_CASE(direct_nn_ff_io)
{
  srand(time(0));
  typedef phen::Parameters<gen::EvoFloat<1, Params>, fit::FitDummy<>, Params> weight_t;
  typedef phen::Parameters<gen::EvoFloat<1, Params>, fit::FitDummy<>, Params> bias_t;
  typedef nn::PfWSum<weight_t> pf_t;
  typedef nn::AfSigmoidBias<bias_t> af_t; 
  typedef nn::Neuron<pf_t, af_t> neuron_t;
  typedef nn::Connection<weight_t> connection_t;
  typedef gen::DnnFF<neuron_t, connection_t, Params> gen_t;
  typedef phen::Dnn<gen_t, fit::FitDummy<Params>, Params> phen_t;
  phen_t i;
  i.random();
  i.develop();
  std::vector<float> in(Params::dnn::nb_inputs, 0);
  for(int i=0;i<Params::dnn::nb_inputs;i++)
    in[i] = misc::rand<float>(0, 1);
  i.nn().step(in);
  float output1 = i.nn().get_neuron_output(0);
  i.show(std::cout);
  
  std::ofstream ofs("/tmp/nn.xml");
  xml_oarchive xml(ofs);
  xml << make_nvp("test", i);
  ofs.close();
  
  
  phen_t* i2 = new phen_t;
  std::ifstream inputFile("/tmp/nn.xml");
  xml_iarchive xml2(inputFile);
  xml2 >> make_nvp("test", *i2);
  inputFile.close();
  
  i2->develop();
  i2->nn().step(in);
  float output2 = i2->nn().get_neuron_output(0);
  i2->show(std::cout);
  
  std::cout << output1 << " " << output2;
  BOOST_CHECK_EQUAL(output1, output2);
}

