using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda.VectorTypes;

namespace Demo
{
    public class ColorMatrix
    {
        private double[] values = new double[9];

        public ColorMatrix()
        {

        }

        public ColorMatrix(double[] v)
        {
            for (int i = 0; i < 9; i++)
            {
                values[i] = v[i];
            }
        }

        public ColorMatrix Invert()
        {

            double a00 = this[0, 0];
            double a01 = this[0, 1];
            double a02 = this[0, 2];
            double a10 = this[1, 0];
            double a11 = this[1, 1];
            double a12 = this[1, 2];
            double a20 = this[2, 0];
            double a21 = this[2, 1];
            double a22 = this[2, 2];

            ColorMatrix temp = new ColorMatrix();

            temp[0, 0] = a11 * a22 - a21 * a12;
            temp[0, 1] = a21 * a02 - a01 * a22;
            temp[0, 2] = a01 * a12 - a11 * a02;
            temp[1, 0] = a20 * a12 - a10 * a22;
            temp[1, 1] = a00 * a22 - a20 * a02;
            temp[1, 2] = a10 * a02 - a00 * a12;
            temp[2, 0] = a10 * a21 - a20 * a11;
            temp[2, 1] = a20 * a01 - a00 * a21;
            temp[2, 2] = a00 * a11 - a10 * a01;

            double det = (a00 * temp[0, 0] +
                          a01 * temp[1, 0] +
                          a02 * temp[2, 0]);

            if (Math.Abs(det) < 0.0000000000000001)
            {
                throw new Exception();
            }

            ColorMatrix B = new ColorMatrix();

            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++)
                {
                    B[j, k] = temp[j, k] / det;
                }

            return B;
        }

        public double this[int x, int y]
        {
            get { return values[x * 3 + y]; }
            set { values[x * 3 + y] = value; }
        }


        #region Operator Methods
        /// <summary>
        /// per element Add
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static ColorMatrix Add(ColorMatrix src, ColorMatrix value)
        {
            double[] val = new double[9];
            for (int i = 0; i < 9; i++)
            {
                val[i] = src.values[i] + value.values[i];
            }
            ColorMatrix ret = new ColorMatrix(val);
            return ret;
        }

        /// <summary>
        /// per element Add
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static ColorMatrix Add(ColorMatrix src, double value)
        {
            double[] val = new double[9];
            for (int i = 0; i < 9; i++)
            {
                val[i] = src.values[i] + value;
            }
            ColorMatrix ret = new ColorMatrix(val);
            return ret;
        }

        /// <summary>
        /// per element Add
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static ColorMatrix Add(double src, ColorMatrix value)
        {
            double[] val = new double[9];
            for (int i = 0; i < 9; i++)
            {
                val[i] = src + value.values[i];
            }
            ColorMatrix ret = new ColorMatrix(val);
            return ret;
        }

        /// <summary>
        /// per element Substract
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static ColorMatrix Subtract(ColorMatrix src, ColorMatrix value)
        {
            double[] val = new double[9];
            for (int i = 0; i < 9; i++)
            {
                val[i] = src.values[i] - value.values[i];
            }
            ColorMatrix ret = new ColorMatrix(val);
            return ret;
        }

        /// <summary>
        /// per element Substract
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static ColorMatrix Subtract(ColorMatrix src, double value)
        {
            double[] val = new double[9];
            for (int i = 0; i < 9; i++)
            {
                val[i] = src.values[i] - value;
            }
            ColorMatrix ret = new ColorMatrix(val);
            return ret;
        }

        /// <summary>
        /// per element Substract
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static ColorMatrix Subtract(double src, ColorMatrix value)
        {
            double[] val = new double[9];
            for (int i = 0; i < 9; i++)
            {
                val[i] = src - value.values[i];
            }
            ColorMatrix ret = new ColorMatrix(val);
            return ret;
        }

        /// <summary>
        /// per element Multiply
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static ColorMatrix Multiply(ColorMatrix src, ColorMatrix value)
        {
            ColorMatrix ret = new ColorMatrix();
            for (int retx = 0; retx < 3; retx++)
                for (int rety = 0; rety < 3; rety++)
                {
                    double val = 0;
                    for (int i = 0; i < 3; i++)
                    {
                        val += src[retx, i] * value[i, rety];
                    }
                    ret[retx, rety] = val;
                }
            return ret;
        }

        /// <summary>
        /// per element Multiply
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static ColorMatrix Multiply(ColorMatrix src, double value)
        {
            double[] val = new double[9];
            for (int i = 0; i < 9; i++)
            {
                val[i] = src.values[i] * value;
            }
            ColorMatrix ret = new ColorMatrix(val);
            return ret;
        }

        /// <summary>
        /// per element Multiply
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static ColorMatrix Multiply(double src, ColorMatrix value)
        {
            double[] val = new double[9];
            for (int i = 0; i < 9; i++)
            {
                val[i] = src * value.values[i];
            }
            ColorMatrix ret = new ColorMatrix(val);
            return ret;
        }


        /// <summary>
        /// per element Divide
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static ColorMatrix Divide(ColorMatrix src, double value)
        {
            double[] val = new double[9];
            for (int i = 0; i < 9; i++)
            {
                val[i] = src.values[i] / value;
            }
            ColorMatrix ret = new ColorMatrix(val);
            return ret;
        }

        /// <summary>
        /// per element Divide
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static ColorMatrix Divide(double src, ColorMatrix value)
        {
            double[] val = new double[9];
            for (int i = 0; i < 9; i++)
            {
                val[i] = src / value.values[i];
            }
            ColorMatrix ret = new ColorMatrix(val);
            return ret;
        }
        #endregion

        #region operators
        /// <summary>
        /// per element
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static ColorMatrix operator +(ColorMatrix src, ColorMatrix value)
        {
            return Add(src, value);
        }

        /// <summary>
        /// per element
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static ColorMatrix operator +(ColorMatrix src, double value)
        {
            return Add(src, value);
        }

        /// <summary>
        /// per element
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static ColorMatrix operator +(double src, ColorMatrix value)
        {
            return Add(src, value);
        }

        /// <summary>
        /// per element
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static ColorMatrix operator -(ColorMatrix src, ColorMatrix value)
        {
            return Subtract(src, value);
        }

        /// <summary>
        /// per element
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static ColorMatrix operator -(ColorMatrix src, double value)
        {
            return Subtract(src, value);
        }

        /// <summary>
        /// per element
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static ColorMatrix operator -(double src, ColorMatrix value)
        {
            return Subtract(src, value);
        }

        /// <summary>
        /// per element
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static ColorMatrix operator *(ColorMatrix src, ColorMatrix value)
        {
            return Multiply(src, value);
        }

        /// <summary>
        /// per element
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static ColorMatrix operator *(ColorMatrix src, double value)
        {
            return Multiply(src, value);
        }

        /// <summary>
        /// per element
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static double3 operator *(ColorMatrix src, double3 value)
        {
            double3 ret = new double3();


            ret.x = src[0, 0] * value.x + src[0, 1] * value.y + src[0, 2] * value.z;
            ret.y = src[1, 0] * value.x + src[1, 1] * value.y + src[1, 2] * value.z;
            ret.z = src[2, 0] * value.x + src[2, 1] * value.y + src[2, 2] * value.z;

            return ret;
        }

        /// <summary>
        /// per element
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static ColorMatrix operator *(double src, ColorMatrix value)
        {
            return Multiply(src, value);
        }

        /// <summary>
        /// per element
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static ColorMatrix operator /(ColorMatrix src, double value)
        {
            return Divide(src, value);
        }

        /// <summary>
        /// per element
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static ColorMatrix operator /(double src, ColorMatrix value)
        {
            return Divide(src, value);
        }

        /// <summary>
        /// per element
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static bool operator ==(ColorMatrix src, ColorMatrix value)
        {
            if (object.ReferenceEquals(src, value)) return true;
            return src.Equals(value);
        }
        /// <summary>
        /// per element
        /// </summary>
        /// <param name="src"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static bool operator !=(ColorMatrix src, ColorMatrix value)
        {
            return !(src == value);
        }
        #endregion

    }
}
