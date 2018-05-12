using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Drawing;
using System.Drawing.Imaging;
using System.Threading.Tasks;


namespace PentaxPefFile
{
    public class DNGFile : FileReader
    {
        public Bitmap mPreview;
        public Bitmap mJpeg;
        ushort[,] mRawImage;
        int mWidth;
        int mHeight;
        int mBayerWidth;
        int mBayerHeight;
        int mBitDepth;
        float[] mMaxVal;
        float[] mMinVal;
        int mISO;
        ExifEntry.ExifCFAPattern.BayerColor[] mBayerPattern;
        IFDDNGWhiteLevel mWhiteLevel;
        IFDDNGBlackLevel mBlackLevel;
        IFDDNGLinearizationTable mLinearizationTable;
        IFDDNGAsShotNeutral mAsShotNeutral;

        public ushort[,] RawImage
        {
            get { return mRawImage; }
        }
        public int RawWidth
        {
            get { return mWidth; }
        }
        public int RawHeight
        {
            get { return mHeight; }
        }
        public int BayerWidth
        {
            get { return mBayerWidth; }
        }
        public int BayerHeight
        {
            get { return mBayerHeight; }
        }
        public ExifEntry.ExifCFAPattern.BayerColor[] BayerPattern
        {
            get { return mBayerPattern; }
        }
        public float[] MaxVal { get { return mMaxVal; } }
        public float[] MinVal { get { return mMinVal; } }
        public int BitDepth { get { return mBitDepth; } }
        public int ISO { get { return mISO; } }
        public float[] AsShotNeutral
        {
            get
            {
                //Some DNG files in the 5k dataset do not contain the mAsShotNeutral IFD: return some default value
                if (mAsShotNeutral == null)
                    return new float[] { 0.5f, 1, 0.5f };

                return new float[] { (float)mAsShotNeutral.Value[0].Value, (float)mAsShotNeutral.Value[1].Value, (float)mAsShotNeutral.Value[2].Value };
            }
        }
        private bool isDNGVersionLarge = false;

        private unsafe class jhead
        {
            public int algo, bits, high, wide, clrs, sraw, psv, restart;
            public int[] vpred;
            public ushort[] quant;
            public ushort[] idct;
            public ushort[][] huff;
            public ushort[][] free;
            public ushort[] row;
            public ushort* rowPtr;

            public jhead()
            {
                algo = 0;
                bits = 0;
                high = 0;
                wide = 0;
                clrs = 0;
                sraw = 0;
                psv = 0;
                restart = 0;

                vpred = new int[6];
                quant = new ushort[64];
                idct = new ushort[64];
                huff = new ushort[20][];
                free = new ushort[20][];
                row = null;
            }
        }

        private unsafe
        ushort[] make_decoder_ref(byte** source)
        {

            int max, len, h, i, j;
            byte* count;
            ushort[] huff;

            count = (*source += 16) - 17;
	        for (max = 16; max > 0 && (count[max] == 0); max--);
            huff = new ushort[1 + (1 << max)];// (ushort*)calloc(1 + (1 << max), sizeof *huff);
	        //merror(huff, "make_decoder()");
	        huff[0] = (ushort)max;
	        for (h = len = 1; len <= max; len++)
		        for (i = 0; i<count[len]; i++, ++*source)
			        for (j = 0; j< 1 << (max - len); j++)
				        if (h <= 1 << max)
					        huff[h++] = (ushort)(len << 8 | **source);
	        return huff;
        }



        unsafe int ljpeg_start(jhead jh, int info_only, Stream s)
        {

            int c, tag, len;
            byte[] data = new byte[0x10000];
            byte* dp;

            //jh = new jhead();
            
            jh.restart = int.MaxValue;

            s.ReadByte();
	        if ((s.ReadByte()) != 0xd8) return 0;
	        do {
                //s.Read(data, 0, 4);
		        if (4 != s.Read(data, 0, 4))
                    return 0;
		        tag = data[0] << 8 | data[1];
		        len = (data[2] << 8 | data[3]) - 2;
		        if (tag <= 0xff00)
                    return 0;

                s.Read(data, 0, len);
                //fread(data, 1, len, ifp);
		        switch (tag) {
		        case 0xffc3:
			        jh.sraw = ((data[7] >> 4) * (data[7] & 15) - 1) & 3;
                    jh.algo = tag & 0xff;
                    jh.bits = data[0];
                    jh.high = data[1] << 8 | data[2];
                    jh.wide = data[3] << 8 | data[4];
                    jh.clrs = data[5] + jh.sraw;
                        //if (len == 9 && !dng_version) getc(ifp);
                        break;
		        case 0xffc1:
		        case 0xffc0:
			        jh.algo = tag & 0xff;
			        jh.bits = data[0];
			        jh.high = data[1] << 8 | data[2];
			        jh.wide = data[3] << 8 | data[4];
			        jh.clrs = data[5] + jh.sraw;
			        //if (len == 9 && !dng_version) getc(ifp);
			        break;
		        case 0xffc4:
                    //if (info_only) break;
                    fixed (byte* ptr = data)
                    {
                        for (dp = ptr; dp < ptr + len && ((c = *dp++) & -20) == 0;)
                            jh.free[c] = jh.huff[c] = make_decoder_ref(&dp);
                    }
			        break;
		        case 0xffda:
			        jh.psv = data[1 + data[0] * 2];
			        jh.bits -= data[3 + data[0] * 2] & 15;
			        break;
		        case 0xffdb:
                    for(c = 0; c < 64; c++)
                        jh.quant[c] = (ushort)(data[c * 2 + 1] << 8 | data[c * 2 + 2]);
			        break;
		        case 0xffdd:
			        jh.restart = data[0] << 8 | data[1];
                    break;
		        }
        } while (tag != 0xffda);

            for (c = 0; c < 19; c++)
                if (jh.huff[c + 1] == null) jh.huff[c + 1] = jh.huff[c];

            //if (jh.sraw) {

            //       FORC(4)        jh->huff[2 + c] = jh->huff[1];

            //       FORC(jh->sraw) jh->huff[1 + c] = jh->huff[0];
            //}
            jh.row = new ushort[jh.wide * jh.clrs * 2]; //(ushort*)calloc(jh->wide* jh->clrs, 4);
	        //merror(jh->row, "ljpeg_start()");
	        return 1;
        }

        private unsafe ushort* ljpeg_row(int jrow, jhead jh, Stream s)
        {
	        int col, c, diff, pred, spred = 0;
            ushort mark = 0;
            ushort*[] row = new ushort*[3];

	        if (jrow * jh.wide % jh.restart == 0)
            {
                for (c = 0; c < 6; c++)
                    jh.vpred[c] = 1 << (jh.bits - 1);

                if (jrow > 0)
                {
                    s.Seek(-2, SeekOrigin.Current);
                    //fseek(ifp, -2, SEEK_CUR);
			        do
                        mark = (ushort)((mark << 8) + (c = s.ReadByte()));
			        while (s.Position < s.Length - 1 && mark >> 4 != 0xffd);
		        }
                getbithuff(-1, null, s);
            }

            for (c = 0; c < 3; c++)
                row[c] = jh.rowPtr + jh.wide * jh.clrs * ((jrow + c) & 1);

	        for (col = 0; col<jh.wide; col++)
                for (c = 0; c < jh.clrs; c++)
                {
                    fixed (ushort* ptr = jh.huff[c])
                    {
                        diff = ljpeg_diff(ptr, s);
                        if (jh.sraw > 0 && c <= jh.sraw && (col | c) > 0)
                            pred = spred;
                        else if (col > 0) pred = row[0][-jh.clrs];
                        else pred = (jh.vpred[c] += diff) - diff;
                        if (jrow > 0 && col > 0)
                            switch (jh.psv)
                            {
                                case 1: break;
                                case 2: pred = row[1][0]; break;
                                case 3: pred = row[1][-jh.clrs]; break;
                                case 4: pred = pred + row[1][0] - row[1][-jh.clrs]; break;
                                case 5: pred = pred + ((row[1][0] - row[1][-jh.clrs]) >> 1); break;
                                case 6: pred = row[1][0] + ((pred - row[1][-jh.clrs]) >> 1); break;
                                case 7: pred = (pred + row[1][0]) >> 1; break;
                                default: pred = 0;
                                    break;
                            }
                        if ((*row[0] = (ushort)(pred + diff)) >> jh.bits != 0)
                            return null;
                        if (c <= jh.sraw) spred = *row[0];
                        row[0]++; row[1]++;
                    }
                }
	            return row[2];
            }

        public DNGFile(string aFileName)
            : base(aFileName)
        {
            byte a = mFileReader.ReadByte();
            byte b = mFileReader.ReadByte();

            bool fileIsLittleEndian;
            if (a == b && b == 'I')
                fileIsLittleEndian = true;
            else
                if (a == b && b == 'M')
                fileIsLittleEndian = false;
            else
                throw new FileLoadException("Could not determine file endianess. Is this a proper TIFF/PEF file?", aFileName);

            mEndianSwap = fileIsLittleEndian != BitConverter.IsLittleEndian;

            ushort magicNumber = ReadUI2();

            if (magicNumber != 42)
                throw new FileLoadException("This is not a valid TIFF/PEF file: Magic number is not 42.", aFileName);

            uint offsetToFirstIFD = ReadUI4();

            mFile.Seek(offsetToFirstIFD, SeekOrigin.Begin);
            List<ImageFileDirectory> ifds = new List<ImageFileDirectory>();
            while (true)
            {
                ImageFileDirectory ifd = new ImageFileDirectory(this);
                ifds.Add(ifd);
                uint offsetToNext = ReadUI4();
                if (offsetToNext == 0)
                    break;
                Seek(offsetToNext, System.IO.SeekOrigin.Begin);
            }

            IFDDNGVersion version = ifds[0].GetEntry<IFDDNGVersion>();
            isDNGVersionLarge = version == null;
            if (version != null)
                isDNGVersionLarge = version.Value[1] > 1;

            IFDSubIFDs sub = ifds[0].GetEntry<IFDSubIFDs>();
            mWidth = (int)sub.Value[0].GetEntry<IFDImageWidth>().Value;
            mHeight = (int)sub.Value[0].GetEntry<IFDImageLength>().Value;
            IFDExif exif = ifds[0].GetEntry<IFDExif>();
            mISO = exif.GetEntry<ExifEntry.ExifISOSpeedRatings>().Value;
            mBitDepth = sub.Value[0].GetEntry<IFDBitsPerSample>().Value;

            mBayerPattern = sub.Value[0].GetEntry<IFDCFAPattern>().Value;
            mBayerWidth = sub.Value[0].GetEntry<IFDCFARepeatPatternDim>().Value[0];
            mBayerHeight = sub.Value[0].GetEntry<IFDCFARepeatPatternDim>().Value[1];
            mWhiteLevel = sub.Value[0].GetEntry<IFDDNGWhiteLevel>();
            mBlackLevel = sub.Value[0].GetEntry<IFDDNGBlackLevel>();

            mAsShotNeutral = sub.Value[0].GetEntry<IFDDNGAsShotNeutral>();

            mLinearizationTable = sub.Value[0].GetEntry<IFDDNGLinearizationTable>();
            mMinVal = new float[3];
            mMinVal[0] = (float)mBlackLevel.Value[0].Value;
            mMinVal[1] = (float)mBlackLevel.Value[mBlackLevel.Value.Length > 1 ? 1 : 0].Value;
            mMinVal[2] = (float)mBlackLevel.Value[mBlackLevel.Value.Length > 3 ? 3 : 0].Value;

            mMaxVal = new float[3];
            mMaxVal[0] = (float)mWhiteLevel.Value[0];
            mMaxVal[1] = (float)mWhiteLevel.Value[mWhiteLevel.Value.Length > 1 ? 1 : 0];
            mMaxVal[2] = (float)mWhiteLevel.Value[mWhiteLevel.Value.Length > 3 ? 3 : 0];

            int tileWidth = (int)sub.Value[0].GetEntry<IFDTileWidth>().Value;
            int tileHeight = (int)sub.Value[0].GetEntry<IFDTileLength>().Value;

            uint[] offsets = sub.Value[0].GetEntry<IFDTileOffsets>().Value;
            uint[] byteCounts = sub.Value[0].GetEntry<IFDTileByteCounts>().Value;

            mRawImage = new ushort[mHeight, mWidth];

            int row = 0;
            int col = 0;
            
            for (int tile = 0; tile < offsets.Length; tile++)
            {
                byte[] data;
                Seek(offsets[tile], SeekOrigin.Begin);
                data = mFileReader.ReadBytes((int)byteCounts[tile]);
                MemoryStream ms = new MemoryStream(data);
                ms.Seek(0, SeekOrigin.Begin);
                jhead jh = new jhead();
                int ret = ljpeg_start(jh, 0, ms);
                int jrow, jcol;
                if (ret > 0 && jh != null)
                {
                    int jwide = jh.wide;
                    jwide *= jh.clrs;

                    
                    unsafe
                    {
                        if (jh.algo == 0xc3) //lossless JPEG
                        {
                            for (jrow = 0; jrow < jh.high; jrow++)
                            {
                                fixed (ushort* ptr = jh.row)
                                {
                                    jh.rowPtr = ptr;
                                    ushort* rp = ljpeg_row(jrow, jh, ms);
                                    for (jcol = 0; jcol < jwide; jcol++)
                                    {
                                        if (jcol + col < mWidth && jrow + row < mHeight)
                                        {
                                            if (mLinearizationTable != null)
                                                mRawImage[row + jrow, col + jcol] = mLinearizationTable.Value[rp[jcol] < mLinearizationTable.Value.Length ? rp[jcol] : mLinearizationTable.Value.Length - 1];
                                            else
                                                mRawImage[row + jrow, col + jcol] = rp[jcol];
                                        }
                                    }
                                    jh.rowPtr = null;
                                }
                            }
                        }
                    }
                }
                col += tileWidth;
                if (col > mWidth)
                {
                    col = 0;
                    row += tileHeight;
                }
            }
                     
            

        }

        public DNGFile(string aFileName, bool HeaderOnly)
            : base(aFileName)
        {
            byte a = mFileReader.ReadByte();
            byte b = mFileReader.ReadByte();

            bool fileIsLittleEndian;
            if (a == b && b == 'I')
                fileIsLittleEndian = true;
            else
                if (a == b && b == 'M')
                fileIsLittleEndian = false;
            else
                throw new FileLoadException("Could not determine file endianess. Is this a proper TIFF/PEF file?", aFileName);

            mEndianSwap = fileIsLittleEndian != BitConverter.IsLittleEndian;

            ushort magicNumber = ReadUI2();

            if (magicNumber != 42)
                throw new FileLoadException("This is not a valid TIFF/PEF file: Magic number is not 42.", aFileName);

            uint offsetToFirstIFD = ReadUI4();

            mFile.Seek(offsetToFirstIFD, SeekOrigin.Begin);
            List<ImageFileDirectory> ifds = new List<ImageFileDirectory>();
            while (true)
            {
                ImageFileDirectory ifd = new ImageFileDirectory(this);
                ifds.Add(ifd);
                uint offsetToNext = ReadUI4();
                if (offsetToNext == 0)
                    break;
                Seek(offsetToNext, System.IO.SeekOrigin.Begin);
            }


            IFDSubIFDs sub = ifds[0].GetEntry<IFDSubIFDs>();
            mWidth = (int)sub.Value[0].GetEntry<IFDImageWidth>().Value;
            mHeight = (int)sub.Value[0].GetEntry<IFDImageLength>().Value;
            IFDExif exif = ifds[0].GetEntry<IFDExif>();
            mISO = exif.GetEntry<ExifEntry.ExifISOSpeedRatings>().Value;
            mBitDepth = sub.Value[0].GetEntry<IFDBitsPerSample>().Value;

            mBayerPattern = sub.Value[0].GetEntry<IFDCFAPattern>().Value;
            mBayerWidth = sub.Value[0].GetEntry<IFDCFARepeatPatternDim>().Value[0];
            mBayerHeight = sub.Value[0].GetEntry<IFDCFARepeatPatternDim>().Value[1];
            mWhiteLevel = sub.Value[0].GetEntry<IFDDNGWhiteLevel>();
            mBlackLevel = sub.Value[0].GetEntry<IFDDNGBlackLevel>();
            mLinearizationTable = sub.Value[0].GetEntry<IFDDNGLinearizationTable>();
            mMinVal = new float[3];
            mMinVal[0] = (float)mBlackLevel.Value[0].Value;
            mMinVal[1] = (float)mBlackLevel.Value[mBlackLevel.Value.Length > 1 ? 1 : 0].Value;
            mMinVal[2] = (float)mBlackLevel.Value[mBlackLevel.Value.Length > 3 ? 3 : 0].Value;

            mMaxVal = new float[3];
            mMaxVal[0] = (float)mWhiteLevel.Value[0];
            mMaxVal[1] = (float)mWhiteLevel.Value[mWhiteLevel.Value.Length > 1 ? 1 : 0];
            mMaxVal[2] = (float)mWhiteLevel.Value[mWhiteLevel.Value.Length > 3 ? 3 : 0];



            mAsShotNeutral = ifds[0].GetEntry<IFDDNGAsShotNeutral>();

        }

        uint bitbuf = 0;
        int vbits = 0, reset = 0;
        private unsafe uint getbithuff(int nbits, ushort* huff, Stream s)
        {
            uint c;

            if (nbits > 25) return 0;
            if (nbits < 0)
            {
                reset = 0;
                vbits = 0;
                bitbuf = 0;
                return 0;
            }

            if (nbits == 0 || vbits < 0) return 0;

            while (!(reset != 0) && vbits < nbits /*&& (c = ReadUI1() fgetc(ifp)) != EOF && !(reset = 0 && c == 0xff && fgetc(ifp))*/)
            {
                c = (uint)s.ReadByte();// ReadUI1();
                if (c == 255)
                {
                    int t = s.ReadByte();
                }
                bitbuf = (bitbuf << 8) + (byte)c;
                vbits += 8;
            }
            c = bitbuf << (32 - vbits) >> (32 - nbits);
            if (huff != null)
            {
                vbits -= huff[c] >> 8;
                c = (byte)huff[c];
            }
            else
                vbits -= nbits;
            //if (vbits < 0) derror();
            return c;
        }

        private unsafe int ljpeg_diff(ushort* huff, Stream s)
        {
            int len, diff;
            len = (int)getbithuff(*huff, huff + 1, s);
            if (len == 16 && isDNGVersionLarge)
                return -32768;

            diff = (int)getbithuff(len, null, s);
            if ((diff & (1 << (len - 1))) == 0)
                diff -= (1 << len) - 1;
            return diff;
        }
    }
}
