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
	public class PEFFile : FileReader
	{
		public Bitmap mPreview;
		public Bitmap mJpeg;
		ushort[,] mRawImage;
		int mWidth;
		int mHeight;
		int mBayerWidth;
		int mBayerHeight;
        int mBitDepth;
        int mISO;
		ExifEntry.ExifCFAPattern.BayerColor[] mBayerPattern;
		MNWhiteLevel mWhiteLevel;
		MNWhitePoint mWhitePoint;
		MNBlackPoint mBlackPoint;
		MNDataScaling mScaling;

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
		public MNWhiteLevel WhiteLevel { get { return mWhiteLevel; } }
		public MNWhitePoint WhitePoint { get { return mWhitePoint; } }
		public MNBlackPoint BlackPoint { get { return mBlackPoint; } }
		public MNDataScaling Scaling { get { return mScaling; } }
        public int BitDepth { get { return mBitDepth; } }
        public int ISO { get { return mISO; } }

        public PEFFile(string aFileName)
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

            //Raw Data:
            ImageFileDirectory raw = ifds[0];
			IFDExif exif = raw.GetEntry<IFDExif>();
            mISO = exif.GetEntry<ExifEntry.ExifISOSpeedRatings>().Value;
            mBitDepth = raw.GetEntry<IFDBitsPerSample>().Value;
			ExifEntry.ExifMakerNote makernote = exif.GetEntry<ExifEntry.ExifMakerNote>();
			mBayerPattern = exif.GetEntry<ExifEntry.ExifCFAPattern>().Value;
			mBayerWidth = exif.GetEntry<ExifEntry.ExifCFAPattern>().xCount;
			mBayerHeight = exif.GetEntry<ExifEntry.ExifCFAPattern>().yCount;
			MNHuffmanTable huffmanTable = makernote.Value.GetEntry<MNHuffmanTable>();
			mWhiteLevel = makernote.Value.GetEntry<MNWhiteLevel>();
			mWhitePoint = makernote.Value.GetEntry<MNWhitePoint>();
			mBlackPoint = makernote.Value.GetEntry<MNBlackPoint>();
			mScaling = makernote.Value.GetEntry<MNDataScaling>(); 

			mWidth = (int)raw.GetEntry<IFDImageWidth>().Value;
			mHeight = (int)raw.GetEntry<IFDImageLength>().Value;
			uint offset = raw.GetEntry<IFDStripOffsets>().Value[0];

			Seek(offset, SeekOrigin.Begin);
			mRawImage = new ushort[mHeight, mWidth];
			int[,] vpred = new int[2,2];
			int[]  hpred = new int[2];

			unsafe
			{
				fixed (ushort* huff = huffmanTable.Value)
				{
					getbithuff(-1, null);

					for (int row = 0; row < mHeight; row++)
						for (int col = 0; col < mWidth; col++)
						{
							int diff = ljpeg_diff(huff);
							if (col < 2)
								hpred[col] = vpred[row & 1, col] += diff;
							else
								hpred[col & 1] += diff;
							mRawImage[row, col] = (ushort)hpred[col & 1];
						}
				}
			}

		}

        public PEFFile(string aFileName, bool HeaderOnly)
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

            //Raw Data:
            ImageFileDirectory raw = ifds[0];
            IFDExif exif = raw.GetEntry<IFDExif>();
            mISO = exif.GetEntry<ExifEntry.ExifISOSpeedRatings>().Value;
            mBitDepth = raw.GetEntry<IFDBitsPerSample>().Value;
            ExifEntry.ExifMakerNote makernote = exif.GetEntry<ExifEntry.ExifMakerNote>();
            mBayerPattern = exif.GetEntry<ExifEntry.ExifCFAPattern>().Value;
            mBayerWidth = exif.GetEntry<ExifEntry.ExifCFAPattern>().xCount;
            mBayerHeight = exif.GetEntry<ExifEntry.ExifCFAPattern>().yCount;
            MNHuffmanTable huffmanTable = makernote.Value.GetEntry<MNHuffmanTable>();
            mWhiteLevel = makernote.Value.GetEntry<MNWhiteLevel>();
            mWhitePoint = makernote.Value.GetEntry<MNWhitePoint>();
            mBlackPoint = makernote.Value.GetEntry<MNBlackPoint>();
            mScaling = makernote.Value.GetEntry<MNDataScaling>();

            mWidth = (int)raw.GetEntry<IFDImageWidth>().Value;
            mHeight = (int)raw.GetEntry<IFDImageLength>().Value;
            uint offset = raw.GetEntry<IFDStripOffsets>().Value[0];
            
        }

        uint bitbuf=0;
		int vbits=0, reset=0;
		private unsafe uint getbithuff (int nbits, ushort *huff)
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
				c = ReadUI1();
				bitbuf = (bitbuf << 8) + (byte) c;
				vbits += 8;
			}
			c = bitbuf << (32-vbits) >> (32-nbits);
			if (huff != null) 
			{
				vbits -= huff[c] >> 8;
				c = (byte) huff[c];
			} 
			else
			vbits -= nbits;
			//if (vbits < 0) derror();
			return c;
		}

		private unsafe int ljpeg_diff(ushort* huff)
		{
			int len, diff;
			len = (int)getbithuff(*huff, huff + 1);
			diff = (int)getbithuff(len, null);
			if ((diff & (1 << (len - 1))) == 0)
				diff -= (1 << len) - 1;
			return diff;
		}
	}
}
