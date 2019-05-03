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


#include <algorithm>
#include <cmath>
#include <iostream>

#include <sferes/parallel.hpp>
#include <sferes/fit/fitness.hpp>
#include <sferes/gen/evo_float.hpp>
#include <sferes/phen/parameters.hpp>
#include <sferes/modif/diversity.hpp>
#include <sferes/ea/nsga2.hpp>
#include <sferes/stat/pareto_front.hpp>
#include <sferes/stat/best_fit.hpp>
#include <sferes/eval/parallel.hpp>
#include <sferes/run.hpp>

#include <modules/nn2/gen_dnn.hpp>
#include <modules/nn2/gen_dnn_ff.hpp>
#include <modules/nn2/phen_dnn.hpp>

using namespace sferes;
using namespace sferes::gen::dnn;
using namespace sferes::gen::evo_float;

struct Params {
    // NEURAL NETWORK PARAMETERS

    // mutations of the weights and biases
    // these parameters are *per weight*
    // (e.g., 0.1 = 0.1 chance for each each weight)
    struct evo_float {
        SFERES_CONST float mutation_rate = 0.1f;
        SFERES_CONST float cross_rate = 0.1f;
        SFERES_CONST mutation_t mutation_type = polynomial;
        SFERES_CONST cross_over_t cross_over_type = sbx;
        SFERES_CONST float eta_m = 15.0f;
        SFERES_CONST float eta_c = 15.0f;
    };
    // minimum and maximum weight/bias
    struct parameters {
        // maximum value of parameters (weights and bias)
        SFERES_CONST float min = -5.0f;
        // minimum value
        SFERES_CONST float max = 5.0f;
    };
    // neural network
    struct dnn {
        // number of inputs (no need for bias here)
        SFERES_CONST size_t nb_inputs = 2;
        // number of outputs
        SFERES_CONST size_t nb_outputs = 1;
        // minimum number of neurons used for the random generation
        SFERES_CONST size_t min_nb_neurons = 2;
        // maximum number of neurons used for the random generation
        SFERES_CONST size_t max_nb_neurons = 5;
        // minimum number of connections used for the random generation
        SFERES_CONST size_t min_nb_conns = 20;
        // maximum number of connections used for the random generation
        SFERES_CONST size_t max_nb_conns = 25;

        // probability to add a connection between two existing neurons
        SFERES_CONST float m_rate_add_conn = 0.05f;
        // probability to remove one connection
        SFERES_CONST float m_rate_del_conn = 0.04f;
        // probability to change the start or the end of an existing connection
        SFERES_CONST float m_rate_change_conn = 0.1f;
        // probability to add a neuron to on an existing connection
        SFERES_CONST float m_rate_add_neuron = 0.025f;
        // probability to delete a neuron (and its connections)
        SFERES_CONST float m_rate_del_neuron = 0.02;

        // do we evolve the bias of the input/output?
        SFERES_CONST int io_param_evolving = true;

        // we initialize with a feed-forward network
        SFERES_CONST init_t init = ff;
    };

    // NSGA-2PARAMETERS
    struct pop {
        // population size
        SFERES_CONST unsigned size = 200;
        // number of generation
        SFERES_CONST unsigned nb_gen = 500;
        // period to write intermediate results
        SFERES_CONST int dump_period = 100;
        // multiplier for initial population size
        SFERES_CONST int initial_aleat = 2;
    };
};

// clang-format off
// FITNESS function
SFERES_FITNESS(FitXOR, sferes::fit::Fitness)
{
public:
    FitXOR() {}
    template <typename Indiv> 
    void eval(Indiv& indiv)
    {
        // we need two objectives: one for the fitness, one for the diversity score
        this->_objs.resize(2);
        _behavior.resize(4);

        float fitness = 0;
        static const std::vector<std::vector<float>> inputs = {{-1, 1,}, {-1, -1}, {1, -1,}, {1, 1}};
        static const std::vector<float> outputs = {-1, 1, -1, 1};
        
        float fit = 0;
        indiv.nn().init();
        for (int i = 0; i < 4 ; ++i)
        {
            // keep in mind that step() performing a single step here
            // but we may have several "layers" and recurrence
            // this is why here we sometimes call it 10 times (this is an arbitrary number)
            // for feed-forward networks (dnn_ff), we can compute the number of required steps
            // and retrieve it by .get_depth() + 1
            for (int j = 0; j < indiv.gen().get_depth() + 1; ++j)
                indiv.nn().step(inputs[i]);
            const std::vector<float> &outf = indiv.nn().get_outf();
            _behavior[i] = outf[0];
            fit -= std::powf(outf[0] - outputs[i], 2.0);
        }
        this->_objs[0] = fit;
        this->_value = fit;
    }

    // behavioral distance for the behavioral diversity modifier
    template<typename Indiv>
    float dist(const Indiv& ind) const {
        assert(_behavior.size() == 4);
        double d = 0;
        for (size_t i = 0; i < _behavior.size(); ++i)
            d += std::powf(_behavior[i] - ind.fit()._behavior[i], 2.0);
        return d;
    }
private:
    // store the behavior
    std::vector<float> _behavior;
};
// clang-format on

int main(int argc, char** argv)
{   
    // FITNESS FUNCTION
    typedef FitXOR<Params> fit_t;

    // NEURAL NETWORK CONFIGURATION
    // type of the weights (no need for fitness here)
    typedef phen::Parameters<gen::EvoFloat<1, Params>, fit::FitDummy<>, Params> weight_t;
    // type of the bias (you can use a different Params class here if needed)
    typedef phen::Parameters<gen::EvoFloat<1, Params>, fit::FitDummy<>, Params> bias_t;
    // the potential function is the weighted sum of inputs
    typedef nn::PfWSum<weight_t> pf_t;
    // the activation functions is tanh(x)
    typedef nn::AfTanh<bias_t> af_t;
    // our neuron type (standard)
    typedef nn::Neuron<pf_t, af_t> neuron_t;
    // our connection type (standard)
    typedef nn::Connection<weight_t> connection_t;
    // for a feed-forward neural network (no recurrence)
    typedef gen::DnnFF<neuron_t, connection_t, Params> gen_t;
    // for a neural network with possible recurrences
    //typedef sferes::gen::Dnn<neuron_t,  connection_t, Params> gen_t;
    // phenotype (developped neural network) -- we need the fitness here
    typedef phen::Dnn<gen_t, fit_t, Params> phen_t;

    // EVOLUTIONARY ALGORITHM CONFIGURATION
    // parallel evaluator
    typedef eval::Parallel<Params> eval_t;
    // list of statistics
    typedef boost::fusion::vector<stat::BestFit<phen_t, Params>, stat::ParetoFront<phen_t, Params>> stat_t;
    // we use a behavioral diversity modifier
    typedef modif::Diversity<phen_t> modifier_t;
    typedef ea::Nsga2<phen_t, eval_t, stat_t, modifier_t, Params> ea_t;
    ea_t ea;

    // RUN THE ALGORITHM
    run_ea(argc, argv, ea);

    return 0;
}
