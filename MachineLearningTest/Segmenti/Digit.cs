using Microsoft.ML.Runtime.Api;

namespace MachineLearningTest.Segmenti
{
    public class Digit
    {
        [Column("0")] public float Up;

        [Column("1")] public float Middle;

        [Column("2")] public float Bottom;

        [Column("3")] public float UpLeft;
        [Column("4")] public float BottomLeft;
        [Column("5")] public float TopRight;
        [Column("6")] public float BottomRight;

        [Column("7")] [ColumnName("Label")]
        public float Label;
    }
}