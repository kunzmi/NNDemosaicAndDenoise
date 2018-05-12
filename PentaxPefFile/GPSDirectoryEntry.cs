using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PentaxPefFile
{
	public class GPSDirectoryEntry
	{
		protected FileReader mPefFile;
		protected ushort mTagID;
		protected TIFFValueType mFieldType;
		protected uint mValueCount;
		protected uint mOffset;

		public GPSDirectoryEntry(FileReader aPefFile)
		{
			mPefFile = aPefFile;
			mTagID = mPefFile.ReadUI2();
			mFieldType = new TIFFValueType((TIFFValueTypes)mPefFile.ReadUI2());
			mValueCount = mPefFile.ReadUI4();
			mOffset = mPefFile.ReadUI4();
		}

		protected GPSDirectoryEntry(FileReader aPefFile, ushort aTagID)
		{
			mPefFile = aPefFile;
			mTagID = aTagID;
			mFieldType = new TIFFValueType((TIFFValueTypes)mPefFile.ReadUI2());
			mValueCount = mPefFile.ReadUI4();
			mOffset = mPefFile.ReadUI4();
		}

		public static GPSDirectoryEntry CreateGPSDirectoryEntry(FileReader aPefFile)
		{
			ushort tagID = aPefFile.ReadUI2();

			switch (tagID)	
			{
				case GPSVersionID.TagID:
					return new GPSVersionID(aPefFile, tagID);
				case GPSLatitudeRef.TagID:
					return new GPSLatitudeRef(aPefFile, tagID);
				case GPSLatitude.TagID:
					return new GPSLatitude(aPefFile, tagID);
				case GPSLongitudeRef.TagID:
					return new GPSLongitudeRef(aPefFile, tagID);
				case GPSLongitude.TagID:
					return new GPSLongitude(aPefFile, tagID);
				case GPSAltitudeRef.TagID:
					return new GPSAltitudeRef(aPefFile, tagID);
				case GPSAltitude.TagID:
					return new GPSAltitude(aPefFile, tagID);
				case GPSTimeStamp.TagID:
					return new GPSTimeStamp(aPefFile, tagID);
				case GPSSatellites.TagID:
					return new GPSSatellites(aPefFile, tagID);
				case GPSStatus.TagID:
					return new GPSStatus(aPefFile, tagID);
				case GPSMeasureMode.TagID:
					return new GPSMeasureMode(aPefFile, tagID);
				case GPSDOP.TagID:
					return new GPSDOP(aPefFile, tagID);
				case GPSSpeedRef.TagID:
					return new GPSSpeedRef(aPefFile, tagID);
				case GPSSpeed.TagID:
					return new GPSSpeed(aPefFile, tagID);
				case GPSTrackRef.TagID:
					return new GPSTrackRef(aPefFile, tagID);
				case GPSTrack.TagID:
					return new GPSTrack(aPefFile, tagID);
				case GPSImgDirectionRef.TagID:
					return new GPSImgDirectionRef(aPefFile, tagID);
				case GPSImgDirection.TagID:
					return new GPSImgDirection(aPefFile, tagID);
				case GPSMapDatum.TagID:
					return new GPSMapDatum(aPefFile, tagID);
				case GPSDestLatitudeRef.TagID:
					return new GPSDestLatitudeRef(aPefFile, tagID);
				case GPSDestLatitude.TagID:
					return new GPSDestLatitude(aPefFile, tagID);
				case GPSDestLongitudeRef.TagID:
					return new GPSDestLongitudeRef(aPefFile, tagID);
				case GPSDestLongitude.TagID:
					return new GPSDestLongitude(aPefFile, tagID);
				case GPSDestBearingRef.TagID:
					return new GPSDestBearingRef(aPefFile, tagID);
				case GPSDestBearing.TagID:
					return new GPSDestBearing(aPefFile, tagID);
				case GPSDestDistanceRef.TagID:
					return new GPSDestDistanceRef(aPefFile, tagID);
				case GPSDestDistance.TagID:
					return new GPSDestDistance(aPefFile, tagID);
				case GPSProcessingMethod.TagID:
					return new GPSProcessingMethod(aPefFile, tagID);
				case GPSAreaInformation.TagID:
					return new GPSAreaInformation(aPefFile, tagID);
				case GPSDateStamp.TagID:
					return new GPSDateStamp(aPefFile, tagID);
				case GPSDifferential.TagID:
					return new GPSDifferential(aPefFile, tagID);
				default:
					return new GPSDirectoryEntry(aPefFile, tagID);
			}
		}
	}
	#region Typed base classes
	public class StringGPSDirectoryEntry : GPSDirectoryEntry
	{
		protected string mValue;

		public StringGPSDirectoryEntry(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			if (mFieldType.SizeInBytes * mValueCount <= 4)
			{
				unsafe
				{
					fixed (uint* ptr = &mOffset)
					{
						byte* bptr = (byte*)ptr;
						byte[] text = new byte[4];

						for (int i = 0; i < mValueCount; i++)
						{
							if (aPefFile.EndianSwap)
								text[i] = bptr[4 / mFieldType.SizeInBytes - i - 1];
							else
								text[i] = bptr[i];
						}

						mValue = ASCIIEncoding.ASCII.GetString(text).Replace("\0", string.Empty);
					}
				}
			}
			else
			{
				uint currentOffset = mPefFile.Position();
				mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

				mValue = mPefFile.ReadStr((int)mFieldType.SizeInBytes * (int)mValueCount).Replace("\0", string.Empty);

				mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
			}
			mValue = mValue.TrimEnd(' ');
		}
	}

	public class ByteGPSDirectoryEntry : GPSDirectoryEntry
	{
		protected byte[] mValue;

		public ByteGPSDirectoryEntry(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			if (mFieldType.SizeInBytes * mValueCount <= 4)
			{
				unsafe
				{
					fixed (uint* ptr = &mOffset)
					{
						byte* ptrUS = (byte*)ptr;
						mValue = new byte[mValueCount];
						for (int i = 0; i < mValueCount; i++)
						{
							if (aPefFile.EndianSwap)
								mValue[i] = ptrUS[4 / mFieldType.SizeInBytes - i - 1];
							else
								mValue[i] = ptrUS[i];
						}
					}
				}
			}
			else
			{
				uint currentOffset = mPefFile.Position();
				mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

				mValue = new byte[mValueCount];
				for (int i = 0; i < mValueCount; i++)
				{
					mValue[i] = mPefFile.ReadUI1();
				}
				mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
			}
		}
	}

	public class SByteGPSDirectoryEntry : GPSDirectoryEntry
	{
		protected sbyte[] mValue;

		public SByteGPSDirectoryEntry(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			if (mFieldType.SizeInBytes * mValueCount <= 4)
			{
				unsafe
				{
					fixed (uint* ptr = &mOffset)
					{
						sbyte* ptrUS = (sbyte*)ptr;
						mValue = new sbyte[mValueCount];
						for (int i = 0; i < mValueCount; i++)
						{
							if (aPefFile.EndianSwap)
								mValue[i] = ptrUS[4 / mFieldType.SizeInBytes - i - 1];
							else
								mValue[i] = ptrUS[i];
						}
					}
				}
			}
			else
			{
				uint currentOffset = mPefFile.Position();
				mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

				mValue = new sbyte[mValueCount];
				for (int i = 0; i < mValueCount; i++)
				{
					mValue[i] = mPefFile.ReadI1();
				}
				mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
			}
		}
	}

	public class UShortGPSDirectoryEntry : GPSDirectoryEntry
	{
		protected ushort[] mValue;

		public UShortGPSDirectoryEntry(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			if (mFieldType.SizeInBytes * mValueCount <= 4)
			{
				unsafe
				{
					fixed (uint* ptr = &mOffset)
					{
						ushort* ptrUS = (ushort*)ptr;
						mValue = new ushort[mValueCount];
						for (int i = 0; i < mValueCount; i++)
						{
							if (aPefFile.EndianSwap)
								mValue[i] = ptrUS[4 / mFieldType.SizeInBytes - i - 1];
							else
								mValue[i] = ptrUS[i];
						}
					}
				}
			}
			else
			{
				uint currentOffset = mPefFile.Position();
				mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

				mValue = new ushort[mValueCount];
				for (int i = 0; i < mValueCount; i++)
				{
					mValue[i] = mPefFile.ReadUI2();
				}
				mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
			}
		}
	}

	public class ShortGPSDirectoryEntry : GPSDirectoryEntry
	{
		protected short[] mValue;

		public ShortGPSDirectoryEntry(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			if (mFieldType.SizeInBytes * mValueCount <= 4)
			{
				unsafe
				{
					fixed (uint* ptr = &mOffset)
					{
						short* ptrUS = (short*)ptr;
						mValue = new short[mValueCount];
						for (int i = 0; i < mValueCount; i++)
						{
							if (aPefFile.EndianSwap)
								mValue[i] = ptrUS[4 / mFieldType.SizeInBytes - i - 1];
							else
								mValue[i] = ptrUS[i];
						}
					}
				}
			}
			else
			{
				uint currentOffset = mPefFile.Position();
				mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

				mValue = new short[mValueCount];
				for (int i = 0; i < mValueCount; i++)
				{
					mValue[i] = mPefFile.ReadI2();
				}
				mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
			}
		}
	}

	public class IntGPSDirectoryEntry : GPSDirectoryEntry
	{
		protected int[] mValue;

		public IntGPSDirectoryEntry(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			if (mFieldType.SizeInBytes * mValueCount <= 4)
			{
				unsafe
				{
					fixed (uint* ptr = &mOffset)
					{
						int* ptrUS = (int*)ptr;
						mValue = new int[mValueCount];
						for (int i = 0; i < mValueCount; i++)
						{
							if (aPefFile.EndianSwap)
								mValue[i] = ptrUS[4 / mFieldType.SizeInBytes - i - 1];
							else
								mValue[i] = ptrUS[i];
						}
					}
				}
			}
			else
			{
				uint currentOffset = mPefFile.Position();
				mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

				mValue = new int[mValueCount];
				for (int i = 0; i < mValueCount; i++)
				{
					mValue[i] = mPefFile.ReadI4();
				}
				mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
			}
		}
	}

	public class UIntGPSDirectoryEntry : GPSDirectoryEntry
	{
		protected uint[] mValue;

		public UIntGPSDirectoryEntry(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			if (mFieldType.SizeInBytes * mValueCount <= 4)
			{
				mValue = new uint[mValueCount];
				mValue[0] = mOffset;
			}
			else
			{
				uint currentOffset = mPefFile.Position();
				mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

				mValue = new uint[mValueCount];
				for (int i = 0; i < mValueCount; i++)
				{
					mValue[i] = mPefFile.ReadUI4();
				}
				mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
			}
		}
	}

	public class RationalGPSDirectoryEntry : GPSDirectoryEntry
	{
		protected Rational[] mValue;

		public RationalGPSDirectoryEntry(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			uint currentOffset = mPefFile.Position();
			mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

			mValue = new Rational[mValueCount];
			for (int i = 0; i < mValueCount; i++)
			{
				uint tempNom = mPefFile.ReadUI4();
				uint tempDenom = mPefFile.ReadUI4();
				mValue[i] = new Rational(tempNom, tempDenom);
			}
			mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
		}
	}

	public class SRationalGPSDirectoryEntry : GPSDirectoryEntry
	{
		protected SRational[] mValue;

		public SRationalGPSDirectoryEntry(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			uint currentOffset = mPefFile.Position();
			mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

			mValue = new SRational[mValueCount];
			for (int i = 0; i < mValueCount; i++)
			{
				int tempNom = mPefFile.ReadI4();
				int tempDenom = mPefFile.ReadI4();
				mValue[i] = new SRational(tempNom, tempDenom);
			}
			mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
		}
	}

	public class FloatGPSDirectoryEntry : GPSDirectoryEntry
	{
		protected float[] mValue;

		public FloatGPSDirectoryEntry(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			if (mFieldType.SizeInBytes * mValueCount <= 4)
			{
				unsafe
				{
					fixed (uint* ptr = &mOffset)
					{
						float* ptrUS = (float*)ptr;
						mValue = new float[mValueCount];
						for (int i = 0; i < mValueCount; i++)
						{
							if (aPefFile.EndianSwap)
								mValue[i] = ptrUS[4 / mFieldType.SizeInBytes - i - 1];
							else
								mValue[i] = ptrUS[i];
						}
					}
				}
			}
			else
			{
				uint currentOffset = mPefFile.Position();
				mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

				mValue = new float[mValueCount];
				for (int i = 0; i < mValueCount; i++)
				{
					mValue[i] = mPefFile.ReadF4();
				}
				mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
			}
		}
	}

	public class DoubleGPSDirectoryEntry : GPSDirectoryEntry
	{
		protected double[] mValue;

		public DoubleGPSDirectoryEntry(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			uint currentOffset = mPefFile.Position();
			mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

			mValue = new double[mValueCount];
			for (int i = 0; i < mValueCount; i++)
			{
				mValue[i] = mPefFile.ReadF8();
			}
			mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
		}
	}
	#endregion

	public class GPSVersionID : ByteGPSDirectoryEntry
	{
		new Version mValue;

		public GPSVersionID(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			if (base.mValue.Length > 3)
				mValue = new Version(base.mValue[0], base.mValue[1], base.mValue[2], base.mValue[3]);
			else if (base.mValue.Length == 3)
				mValue = new Version(base.mValue[0], base.mValue[1], base.mValue[2]);
			else if (base.mValue.Length == 2)
				mValue = new Version(base.mValue[0], base.mValue[1]);
			else if (base.mValue.Length == 1)
				mValue = new Version(base.mValue[0], 0);
			else
				mValue = new Version();
		}

		public const ushort TagID = 0;

		public const string TagName = "Version ID";

		public Version Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class GPSLatitudeRef : StringGPSDirectoryEntry
	{
		public GPSLatitudeRef(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 1;

		public const string TagName = "Latitude reference";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class GPSLatitude : RationalGPSDirectoryEntry
	{
		public GPSLatitude(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 2;

		public const string TagName = "Latitude";

		public Rational[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value[0].ToString() + "; " + Value[1].ToString() + "; " + Value[2].ToString();
		}
	}

	public class GPSLongitudeRef : StringGPSDirectoryEntry
	{
		public GPSLongitudeRef(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 3;

		public const string TagName = "Longitude reference";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class GPSLongitude : RationalGPSDirectoryEntry
	{
		public GPSLongitude(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 4;

		public const string TagName = "Longitude";

		public Rational[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value[0].ToString() + "; " + Value[1].ToString() + "; " + Value[2].ToString();
		}
	}

	public class GPSAltitudeRef : ByteGPSDirectoryEntry
	{
		public enum AltitudeRef
		{ 
			AboveSeaLevel = 0,
			BelowSeaLevel = 1
		}

		public GPSAltitudeRef(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 5;

		public const string TagName = "Altitude reference";

		public AltitudeRef Value
		{
			get { return (AltitudeRef)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class GPSAltitude : RationalGPSDirectoryEntry
	{
		public GPSAltitude(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 6;

		public const string TagName = "Altitude";

		public Rational Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class GPSTimeStamp : RationalGPSDirectoryEntry
	{
		public GPSTimeStamp(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 7;

		public const string TagName = "Time stamp";

		public Rational[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value[0].ToString() + "; " + Value[1].ToString() + "; " + Value[2].ToString();
		}
	}

	public class GPSSatellites : StringGPSDirectoryEntry
	{
		public GPSSatellites(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 8;

		public const string TagName = "Satellites";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class GPSStatus : StringGPSDirectoryEntry
	{
		public GPSStatus(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 9;

		public const string TagName = "Status";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class GPSMeasureMode : StringGPSDirectoryEntry
	{
		public GPSMeasureMode(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 10;

		public const string TagName = "Measure mode";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class GPSDOP : RationalGPSDirectoryEntry
	{
		public GPSDOP(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 11;

		public const string TagName = "DOP (data degree of precision)";

		public Rational Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class GPSSpeedRef : StringGPSDirectoryEntry
	{
		public GPSSpeedRef(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 12;

		public const string TagName = "Speed reference";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class GPSSpeed : RationalGPSDirectoryEntry
	{
		public GPSSpeed(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 13;

		public const string TagName = "Speed";

		public Rational Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class GPSTrackRef : StringGPSDirectoryEntry
	{
		public GPSTrackRef(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 14;

		public const string TagName = "Track reference";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class GPSTrack : RationalGPSDirectoryEntry
	{
		public GPSTrack(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 15;

		public const string TagName = "Track";

		public Rational Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class GPSImgDirectionRef : StringGPSDirectoryEntry
	{
		public GPSImgDirectionRef(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 16;

		public const string TagName = "Imgage direction reference";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class GPSImgDirection : RationalGPSDirectoryEntry
	{
		public GPSImgDirection(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 17;

		public const string TagName = "Imgage direction";

		public Rational Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class GPSMapDatum : StringGPSDirectoryEntry
	{
		public GPSMapDatum(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 18;

		public const string TagName = "Map datum";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class GPSDestLatitudeRef : StringGPSDirectoryEntry
	{
		public GPSDestLatitudeRef(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 19;

		public const string TagName = "Destination latitude reference";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class GPSDestLatitude : RationalGPSDirectoryEntry
	{
		public GPSDestLatitude(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 20;

		public const string TagName = "Destination latitude";

		public Rational[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value[0].ToString() + "; " + Value[1].ToString() + "; " + Value[2].ToString();
		}
	}

	public class GPSDestLongitudeRef : StringGPSDirectoryEntry
	{
		public GPSDestLongitudeRef(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 21;

		public const string TagName = "Destination longitude reference";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class GPSDestLongitude : RationalGPSDirectoryEntry
	{
		public GPSDestLongitude(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 22;

		public const string TagName = "Destination longitude";

		public Rational[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value[0].ToString() + "; " + Value[1].ToString() + "; " + Value[2].ToString();
		}
	}

	public class GPSDestBearingRef : StringGPSDirectoryEntry
	{
		public GPSDestBearingRef(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 23;

		public const string TagName = "Destination bearing reference";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class GPSDestBearing : RationalGPSDirectoryEntry
	{
		public GPSDestBearing(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 24;

		public const string TagName = "Destination bearing";

		public Rational Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class GPSDestDistanceRef : StringGPSDirectoryEntry
	{
		public GPSDestDistanceRef(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 25;

		public const string TagName = "Destination distance reference";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class GPSDestDistance : RationalGPSDirectoryEntry
	{
		public GPSDestDistance(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 26;

		public const string TagName = "Destination distance";

		public Rational Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class GPSProcessingMethod : StringGPSDirectoryEntry
	{
		public GPSProcessingMethod(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 27;

		public const string TagName = "Processing method";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class GPSAreaInformation : StringGPSDirectoryEntry
	{
		public GPSAreaInformation(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 28;

		public const string TagName = "Area information";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class GPSDateStamp : StringGPSDirectoryEntry
	{
		DateTime dt_value;

		public GPSDateStamp(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			int year = int.Parse(mValue.Substring(0, 4));
			int month = int.Parse(mValue.Substring(5, 2));
			int day = int.Parse(mValue.Substring(8, 2));
			dt_value = new DateTime(year, month, day);
		}

		public const ushort TagID = 29;

		public const string TagName = "Date stamp";

		public DateTime Value
		{
			get { return dt_value; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}

	public class GPSDifferential : UShortGPSDirectoryEntry
	{
		bool b_value;

		public GPSDifferential(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			b_value = mValue[0] == 1;
		}

		public const ushort TagID = 30;

		public const string TagName = "Differential";

		public bool Value
		{
			get { return b_value; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value;
		}
	}
}
