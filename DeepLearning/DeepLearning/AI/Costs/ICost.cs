using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearning.AI.Costs
{
    public interface ICost
    {
        public double Cost(double predicted, double expected);
        public double CostDerivative(double predicted, double expected);
    }
}
