using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearning.AI.Activations
{
    public static class Activation
    {
        public enum ActivationFunc
        {
            TanH,
        }

        public static IActivation GetActivationFunction(ActivationFunc func)
        {
            switch (func)
            {
                case ActivationFunc.TanH:
                    return new TanH();
                default:
                    Console.WriteLine("Bad activation type");
                    return new TanH();
            }

        }


        public readonly struct TanH : IActivation
        {
            public double Activation(double weightedInput)
            {
                throw new NotImplementedException();
            }

            public double Derivative(double weightedInput)
            {
                throw new NotImplementedException();
            }
        }
    }
}
