using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearning.AI
{
    public class LayerLearnData
    {
        public double[] weightedInput;
        public double[] neuron;
        public double[] nodeValues;
        public double[,] Wgradient;
        public double[] Bgradient;

        public LayerLearnData(double[,] wgradient, double[] bgradient)
        {
            this.weightedInput = new double[bgradient.Length];
            this.neuron = new double[bgradient.Length];
            this.nodeValues = new double[bgradient.Length];
            this.Wgradient = wgradient;
            this.Bgradient = bgradient;
        }
    }
}
