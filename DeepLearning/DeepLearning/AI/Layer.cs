using DeepLearning.AI.Activations;
using DeepLearning.AI.Costs;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearning.AI
{
    public class Layer
    {
        public int size = 0;
        public int previousSize = 0;

        public object GradientLock = new object();

        public double[,] weights;
        public double[,] weightVelocities;
        public double[,] JoinedWgradient;

        public double[] bias;
        public double[] biasVelocities;
        public double[] JoinedBgradient;

        IActivation activation;
        ICost costFunction;

        public Layer(int size, int previousSize, Activation.ActivationFunc activFunc, Cost.CostFunc costFunc)
        {
            this.size = size;
            this.previousSize = previousSize;

            activation = Activation.GetActivationFunction(activFunc);
            costFunction = Cost.GetCostFunction(costFunc);

            Random rand = new Random();

            JoinedBgradient = new double[size];
            biasVelocities = new double[size];
            bias = new double[size];

            weights = new double[previousSize, size];
            weightVelocities = new double[previousSize, size];
            JoinedWgradient = new double[previousSize, size];

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < previousSize; j++)
                {
                    weights[j, i] = rand.NextDouble() * 2 - 1;
                }

                bias[i] = rand.NextDouble() * 2 - 1;
            }
        }

        public void FeedForward(double[] inputs, LayerLearnData dat)
        {
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < previousSize; j++)
                {
                    dat.weightedInput[i] += inputs[j] * weights[j, i];
                }

                dat.weightedInput[i] += bias[i];

                dat.neuron[i] = activation.Activation(dat.weightedInput[i]);
            }
        }

        public void CalculateOutputLayerNodeValues(double[] expectedValues, LayerLearnData dat)
        {
            for (int i = 0; i < size; i++)
            {
                dat.nodeValues[i] = activation.Derivative(dat.weightedInput[i]) * costFunction.CostDerivative(dat.neuron[i], expectedValues[i]);
            }
        }

        public void CalculateGradients(double[] neuronsPrevious, LayerLearnData dat)
        {
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < previousSize; j++)
                {
                    dat.Wgradient[j, i] += neuronsPrevious[j] * dat.nodeValues[i];
                }

                dat.Bgradient[i] += dat.nodeValues[i];
            }
        }

        public void CalculateNodeValues(Layer oldLayerL, LayerLearnData oldLayerD, LayerLearnData dat)
        {
            for (int i = 0; i < size; i++)
            {
                double value = 0;
                for (int j = 0; j < oldLayerD.nodeValues.Length; j++)
                {
                    value += oldLayerD.nodeValues[j] * oldLayerL.weights[i, j];
                }
                dat.nodeValues[i] = value * activation.Derivative(dat.weightedInput[i]);
            }
        }

        /*
        public void ApplyGradients(double learnrateOverbatchSize, double regularization, double momentum)
        {
            double weightDecay = 1 - regularization * learnrateOverbatchSize;

            for (int i = 0; i < weights.GetLength(0); i++)
            {
                for (int j = 0; j < weights.GetLength(1); j++)
                {
                    double weight = weights[i, j];
                    double velocity = weightVelocities[i, j] * momentum - JoinedWgradient[i, j] * learnrateOverbatchSize;
                    weightVelocities[i, j] = velocity;

                    weights[i, j] = weight * weightDecay + velocity;

                    JoinedWgradient[i, j] = 0;
                }
            }


            for (int i = 0; i < bias.Length; i++)
            {
                double velocity = biasVelocities[i] * momentum - JoinedBgradient[i] * learnrateOverbatchSize;
                biasVelocities[i] = velocity;

                bias[i] += velocity;

                JoinedBgradient[i] = 0;
            }
        }*/

        public void ClearGradient()
        {
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < previousSize; j++)
                {
                    JoinedWgradient[j, i] = 0;
                    weightVelocities[j, i] = 0;
                }

                JoinedBgradient[i] = 0;
                biasVelocities[i] = 0;
            }
        }
    }
}
