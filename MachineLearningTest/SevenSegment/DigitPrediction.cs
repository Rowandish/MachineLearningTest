using Microsoft.ML.Runtime.Api;

namespace MachineLearningTest.SevenSegment
{
    public class DigitPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint ExpectedDigit;

        [ColumnName("Score")] public float[] Score;
    }
}
