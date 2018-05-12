using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda.VectorTypes;

namespace Demo
{
    public class ColorManagment
    {
        public struct ruvt
        {
            public double r;
            public double u;
            public double v;
            public double t;

            public ruvt(double _r, double _u, double _v, double _t)
            {
                r = _r;
                u = _u;
                v = _v;
                t = _t;
            }
        }

        public static ruvt[] kTempTable = new ruvt[]
        {
            new ruvt(   0, 0.18006, 0.26352, -0.24341 ),
            new ruvt(  10, 0.18066, 0.26589, -0.25479 ),
            new ruvt(  20, 0.18133, 0.26846, -0.26876 ),
            new ruvt(  30, 0.18208, 0.27119, -0.28539 ),
            new ruvt(  40, 0.18293, 0.27407, -0.30470 ),
            new ruvt(  50, 0.18388, 0.27709, -0.32675 ),
            new ruvt(  60, 0.18494, 0.28021, -0.35156 ),
            new ruvt(  70, 0.18611, 0.28342, -0.37915 ),
            new ruvt(  80, 0.18740, 0.28668, -0.40955 ),
            new ruvt(  90, 0.18880, 0.28997, -0.44278 ),
            new ruvt( 100, 0.19032, 0.29326, -0.47888 ),
            new ruvt( 125, 0.19462, 0.30141, -0.58204 ),
            new ruvt( 150, 0.19962, 0.30921, -0.70471 ),
            new ruvt( 175, 0.20525, 0.31647, -0.84901 ),
            new ruvt( 200, 0.21142, 0.32312, -1.0182 ),
            new ruvt( 225, 0.21807, 0.32909, -1.2168 ),
            new ruvt( 250, 0.22511, 0.33439, -1.4512 ),
            new ruvt( 275, 0.23247, 0.33904, -1.7298 ),
            new ruvt( 300, 0.24010, 0.34308, -2.0637 ),
            new ruvt( 325, 0.24702, 0.34655, -2.4681 ),
            new ruvt( 350, 0.25591, 0.34951, -2.9641 ),
            new ruvt( 375, 0.26400, 0.35200, -3.5814 ),
            new ruvt( 400, 0.27218, 0.35407, -4.3633 ),
            new ruvt( 425, 0.28039, 0.35577, -5.3762 ),
            new ruvt( 450, 0.28863, 0.35714, -6.7262 ),
            new ruvt( 475, 0.29685, 0.35823, -8.5955 ),
            new ruvt( 500, 0.30505, 0.35907, -11.324 ),
            new ruvt( 525, 0.31320, 0.35968, -15.628 ),
            new ruvt( 550, 0.32129, 0.36011, -23.325 ),
            new ruvt( 575, 0.32931, 0.36038, -40.770 ),
            new ruvt( 600, 0.33724, 0.36051, -116.45 )
        };

        public class dng_temperature
        {
            private double fTemperature;
            private double fTint;

            const double kTintScale = -3000.0;

            public dng_temperature(double2 xy)
            {
                // Convert to uv space.

                double u = 2.0 * xy.x / (1.5 - xy.x + 6.0 * xy.y);
                double v = 3.0 * xy.y / (1.5 - xy.x + 6.0 * xy.y);

                // Search for line pair coordinate is between.

                double last_dt = 0.0;

                double last_dv = 0.0;
                double last_du = 0.0;

                for (uint index = 1; index <= 30; index++)
                {

                    // Convert slope to delta-u and delta-v, with length 1.

                    double du = 1.0;
                    double dv = kTempTable[index].t;

                    double len = Math.Sqrt(1.0 + dv * dv);

                    du /= len;
                    dv /= len;

                    // Find delta from black body point to test coordinate.

                    double uu = u - kTempTable[index].u;
                    double vv = v - kTempTable[index].v;

                    // Find distance above or below line.

                    double dt = -uu * dv + vv * du;

                    // If below line, we have found line pair.

                    if (dt <= 0.0 || index == 30)
                    {

                        // Find fractional weight of two lines.

                        if (dt > 0.0)
                            dt = 0.0;

                        dt = -dt;

                        double f;

                        if (index == 1)
                        {
                            f = 0.0;
                        }
                        else
                        {
                            f = dt / (last_dt + dt);
                        }

                        // Interpolate the temperature.

                        fTemperature = 1.0E6 / (kTempTable[index - 1].r * f +
                                                kTempTable[index].r * (1.0 - f));

                        // Find delta from black body point to test coordinate.

                        uu = u - (kTempTable[index - 1].u * f +
                                  kTempTable[index].u * (1.0 - f));

                        vv = v - (kTempTable[index - 1].v * f +
                                  kTempTable[index].v * (1.0 - f));

                        // Interpolate vectors along slope.

                        du = du * (1.0 - f) + last_du * f;
                        dv = dv * (1.0 - f) + last_dv * f;

                        len = Math.Sqrt(du * du + dv * dv);

                        du /= len;
                        dv /= len;

                        // Find distance along slope.

                        fTint = (uu * du + vv * dv) * kTintScale;

                        break;

                    }

                    // Try next line pair.

                    last_dt = dt;

                    last_du = du;
                    last_dv = dv;

                }

            }

            public double Temperature
            {
                get { return fTemperature; }
            }

            public double Tint
            {
                get { return fTint; }
            }
        }

        private const double fTemperature1 = 2856; // CIE Illuminant A --> reference for matrix1
        private const double fTemperature2 = 6504; //CIE Illuminant D65 --> reference for matrix2

        //extracted from DNG
        private ColorMatrix fColorMatrix1 = new ColorMatrix(new double[] { 0.9281921387, -0.4597167969, -0.07611083984, -0.3495941162, 0.9834136963, 0.4286651611, -0.02061462402, 0.0431060791, 0.788482666 });
        private ColorMatrix fColorMatrix2 = new ColorMatrix(new double[] { 0.8190460205, -0.2475128174, -0.1096954346, -0.3995361328, 1.230117798, 0.1880645752, -0.1049804687, 0.1840820313, 0.6998596191 });

        public double2 StdA_xy_coord
        {
            get { return new double2(0.4476, 0.4074); }
        }
        public double2 D50_xy_coord
        {
            get { return new double2(0.3457, 0.3585); }
        }
        public double2 D55_xy_coord
        {
            get { return new double2(0.3324, 0.3474); }
        }
        public double2 D65_xy_coord
        {
            get { return new double2(0.3127, 0.3290); }
        }
        public double2 D75_xy_coord
        {
            get { return new double2(0.2990, 0.3149); }
        }

        public double2 XYZtoXY(double3 coord)
        {
            double X = coord.x;
            double Y = coord.y;
            double Z = coord.z;

            double total = X + Y + Z;

            if (total > 0.0)
            {
                return new double2(X / total, Y / total);
            }

            return D50_xy_coord;
        }

        private double clamp(double min, double x, double max)
        {
            return Math.Max(min, Math.Min(x, max));
        }

        private double clamp(double x)
        {
            return clamp(0, x, 1);
        }

        double2 fWhiteXY;
        double3 fCameraWhite;
        int fChannels = 3;
        ColorMatrix fPCStoCamera;
        ColorMatrix fCameraToPCS;

        public double2 WhiteXY
        {
            get { return fWhiteXY; }
        }
        public double3 CameraWhite
        {
            get { return fCameraWhite; }
        }
        public ColorMatrix PCStoCamera
        {
            get { return fPCStoCamera; }
        }
        public ColorMatrix CameraToPCS
        {
            get { return fCameraToPCS; }
        }

        public void SetWhiteXY(double2 white)
        {
            fWhiteXY = white;


            // Interpolate an matric values for this white point.

            ColorMatrix colorMatrix;

            colorMatrix = FindXYZtoCamera(fWhiteXY);

            // Find the camera white values.

            fCameraWhite = colorMatrix * XYtoXYZ(fWhiteXY);

            double whiteScale = 1.0 / Math.Max(fCameraWhite.x, Math.Max(fCameraWhite.y, fCameraWhite.z));



            // We don't support non-positive values for camera neutral values.

            fCameraWhite.x = clamp(0.001, whiteScale * fCameraWhite.x, 1.0);
            fCameraWhite.y = clamp(0.001, whiteScale * fCameraWhite.y, 1.0);
            fCameraWhite.z = clamp(0.001, whiteScale * fCameraWhite.z, 1.0);


            // Find PCS to Camera transform. Scale matrix so PCS white can just be
            // reached when the first camera channel saturates

            fPCStoCamera = colorMatrix * MapWhiteMatrix(D50_xy_coord, fWhiteXY);

            double3 tempVal = fPCStoCamera * XYtoXYZ(D50_xy_coord);
            double scale = Math.Max(tempVal.x, Math.Max(tempVal.y, tempVal.z));

            fPCStoCamera = (1.0 / scale) * fPCStoCamera;


            // we need to use the adapt in XYZ method.



            // Invert this PCS to camera matrix.  Note that if there are more than three
            // camera channels, this inversion is non-unique.

            fCameraToPCS = fPCStoCamera.Invert();


        }

        public double3 XYtoXYZ(double2 coord)
        {


            double2 temp = coord;

            // Restrict xy coord to someplace inside the range of real xy coordinates.
            // This prevents math from doing strange things when users specify
            // extreme temperature/tint coordinates.

            temp.x = clamp(0.000001, temp.x, 0.999999);
            temp.y = clamp(0.000001, temp.y, 0.999999);

            if (temp.x + temp.y > 0.999999)
            {
                double scale = 0.999999 / (temp.x + temp.y);
                temp.x *= scale;
                temp.y *= scale;
            }

            return new double3(temp.x / temp.y,
                                 1.0,
                                 (1.0 - temp.x - temp.y) / temp.y);
        }

        public ColorMatrix FindXYZtoCamera(double2 white)
        {
            // Convert to temperature/offset space.

            dng_temperature td = new dng_temperature(white);

            // Find fraction to weight the first calibration.

            double g;

            if (td.Temperature <= fTemperature1)
                g = 1.0;

            else if (td.Temperature >= fTemperature2)
                g = 0.0;

            else
            {

                double invT = 1.0 / td.Temperature;

                g = (invT - (1.0 / fTemperature2)) /
                    ((1.0 / fTemperature1) - (1.0 / fTemperature2));

            }

            // Interpolate the color matrix.

            ColorMatrix colorMatrix;

            if (g >= 1.0)
                colorMatrix = fColorMatrix1;

            else if (g <= 0.0)
                colorMatrix = fColorMatrix2;

            else
                colorMatrix = (g) * fColorMatrix1 +
                              (1.0 - g) * fColorMatrix2;

            // Return the interpolated color matrix.

            return colorMatrix;
        }

        public double2 NeutralToXY(double3 neutral)
        {
            const uint kMaxPasses = 30;

            double2 last = D50_xy_coord;

            for (uint pass = 0; pass < kMaxPasses; pass++)
            {

                ColorMatrix xyzToCamera = FindXYZtoCamera(last);

                double2 next = XYZtoXY((xyzToCamera.Invert()) * neutral);

                if (Math.Abs(next.x - last.x) +
                    Math.Abs(next.y - last.y) < 0.0000001)
                {

                    return next;

                }

                // If we reach the limit without converging, we are most likely
                // in a two value oscillation.  So take the average of the last
                // two estimates and give up.

                if (pass == kMaxPasses - 1)
                {

                    next.x = (last.x + next.x) * 0.5;
                    next.y = (last.y + next.y) * 0.5;

                }

                last = next;

            }

            return last;

        }

        public ColorMatrix MapWhiteMatrix(double2 white1, double2 white2)
        {
            // Use the linearized Bradford adaptation matrix.

            ColorMatrix Mb = new ColorMatrix(new double[]{
                0.8951,  0.2664, -0.1614,
               -0.7502,  1.7135,  0.0367,
                0.0389, -0.0685,  1.0296});

            double3 w1 = Mb * XYtoXYZ(white1);
            double3 w2 = Mb * XYtoXYZ(white2);

            // Negative white coordinates are kind of meaningless.

            w1.x = Math.Max(w1.x, 0.0);
            w1.y = Math.Max(w1.y, 0.0);
            w1.z = Math.Max(w1.z, 0.0);

            w2.x = Math.Max(w2.x, 0.0);
            w2.y = Math.Max(w2.y, 0.0);
            w2.z = Math.Max(w2.z, 0.0);

            // Limit scaling to something reasonable.

            ColorMatrix A = new ColorMatrix();

            A[0, 0] = clamp(0.1, w1.x > 0.0 ? w2.x / w1.x : 10.0, 10.0);
            A[1, 1] = clamp(0.1, w1.y > 0.0 ? w2.y / w1.y : 10.0, 10.0);
            A[2, 2] = clamp(0.1, w1.z > 0.0 ? w2.z / w1.z : 10.0, 10.0);

            ColorMatrix B = (Mb.Invert()) * A * Mb;

            return B;
        }
    }
}
