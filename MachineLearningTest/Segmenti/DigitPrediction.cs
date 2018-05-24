using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Runtime.Api;

namespace MachineLearningTest.Segmenti
{
    public class DigitPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint ExpectedDigit;

        [ColumnName("Score")] public float[] Score;
    }
}
