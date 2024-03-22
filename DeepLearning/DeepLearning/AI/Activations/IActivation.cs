using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearning.AI.Activations
{
    public interface IActivation
    {
        public double Activation(double weightedInput);
        public double Derivative(double weightedInput);
    }
}
