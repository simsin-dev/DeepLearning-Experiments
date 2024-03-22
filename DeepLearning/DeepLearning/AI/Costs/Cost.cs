using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearning.AI.Costs
{
    public static class Cost
    {
        public enum CostFunc
        {
            MSE
        }

        public static ICost GetCostFunction(CostFunc func) 
        {
            switch (func) 
            {
                case CostFunc.MSE:
                    return new MSE();
                default:
                    Console.WriteLine("Bad cost type");
                    return new MSE();
            }
        }

        public readonly struct MSE : ICost
        {
            public double Cost(double predicted, double expected)
            {
                double err = predicted - expected;
                return err * err;
            }

            public double CostDerivative(double predicted, double expected)
            {
                return 2 * (predicted - expected);
            }
        }
    }
}
