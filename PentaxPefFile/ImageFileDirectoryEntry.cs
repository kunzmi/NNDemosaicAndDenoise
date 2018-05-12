using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PentaxPefFile
{
	public class ImageFileDirectoryEntry
	{
		protected FileReader mPefFile;
		protected ushort mTagID;
		protected TIFFValueType mFieldType;
		protected uint mValueCount;
		protected uint mOffset;

		public ImageFileDirectoryEntry(FileReader aPefFile)
		{
			mPefFile = aPefFile;
			mTagID = mPefFile.ReadUI2();
			mFieldType = new TIFFValueType((TIFFValueTypes)mPefFile.ReadUI2());
			mValueCount = mPefFile.ReadUI4();
			mOffset = mPefFile.ReadUI4();
		}

		protected ImageFileDirectoryEntry(FileReader aPefFile, ushort aTagID)
		{
			mPefFile = aPefFile;
			mTagID = aTagID;
			mFieldType = new TIFFValueType((TIFFValueTypes)mPefFile.ReadUI2());
			mValueCount = mPefFile.ReadUI4();
			mOffset = mPefFile.ReadUI4();
		}

		public static ImageFileDirectoryEntry CreateImageFileDirectoryEntry(FileReader aPefFile)
		{
			ushort tagID = aPefFile.ReadUI2();

			switch (tagID)	
			{
				case IFDArtist.TagID: 
					return new IFDArtist(aPefFile, tagID);
				case IFDBitsPerSample.TagID:
					return new IFDBitsPerSample(aPefFile, tagID);
				case IFDCellLength.TagID:
					return new IFDCellLength(aPefFile, tagID);
				case IFDCellWidth.TagID:
					return new IFDCellWidth(aPefFile, tagID);
				case IFDColorMap.TagID:
					return new IFDColorMap(aPefFile, tagID);
				case IFDCompression.TagID:
					return new IFDCompression(aPefFile, tagID);
				case IFDCopyright.TagID:
					return new IFDCopyright(aPefFile, tagID);
				case IFDDateTime.TagID:
					return new IFDDateTime(aPefFile, tagID);
				case IFDExtraSamples.TagID:
					return new IFDExtraSamples(aPefFile, tagID);
				case IFDFillOrder.TagID:
					return new IFDFillOrder(aPefFile, tagID);
				case IFDFreeByteCounts.TagID:
					return new IFDFreeByteCounts(aPefFile, tagID);
				case IFDFreeOffsets.TagID:
					return new IFDFreeOffsets(aPefFile, tagID);
				case IFDGrayResponseCurve.TagID:
					return new IFDGrayResponseCurve(aPefFile, tagID);
				case IFDGrayResponseUnit.TagID:
					return new IFDGrayResponseUnit(aPefFile, tagID);
				case IFDHostComputer.TagID:
					return new IFDHostComputer(aPefFile, tagID);
				case IFDImageDescription.TagID:
					return new IFDImageDescription(aPefFile, tagID);
				case IFDImageLength.TagID:
					return new IFDImageLength(aPefFile, tagID);
				case IFDImageWidth.TagID:
					return new IFDImageWidth(aPefFile, tagID);
				case IFDMake.TagID:
					return new IFDMake(aPefFile, tagID);
				case IFDMaxSampleValue.TagID:
					return new IFDMaxSampleValue(aPefFile, tagID);
				case IFDMinSampleValue.TagID:
					return new IFDMinSampleValue(aPefFile, tagID);
				case IFDModel.TagID:
					return new IFDModel(aPefFile, tagID);
				case IFDNewSubfileType.TagID:
					return new IFDNewSubfileType(aPefFile, tagID);
				case IFDOrientation.TagID:
					return new IFDOrientation(aPefFile, tagID);
				case IFDPhotometricInterpretation.TagID:
					return new IFDPhotometricInterpretation(aPefFile, tagID);
				case IFDPlanarConfiguration.TagID:
					return new IFDPlanarConfiguration(aPefFile, tagID);
				case IFDResolutionUnit.TagID:
					return new IFDResolutionUnit(aPefFile, tagID);
				case IFDRowsPerStrip.TagID:
					return new IFDRowsPerStrip(aPefFile, tagID);
				case IFDSamplesPerPixel.TagID:
					return new IFDSamplesPerPixel(aPefFile, tagID);
				case IFDSoftware.TagID:
					return new IFDSoftware(aPefFile, tagID);
				case IFDStripByteCounts.TagID:
					return new IFDStripByteCounts(aPefFile, tagID);
				case IFDStripOffsets.TagID:
					return new IFDStripOffsets(aPefFile, tagID);
				case IFDSubfileType.TagID:
					return new IFDSubfileType(aPefFile, tagID);
				case IFDThreshholding.TagID:
					return new IFDThreshholding(aPefFile, tagID);
				case IFDXResolution.TagID:
					return new IFDXResolution(aPefFile, tagID);
				case IFDYResolution.TagID:
					return new IFDYResolution(aPefFile, tagID);
				case IFDExif.TagID:
					return new IFDExif(aPefFile, tagID);
				case IFDGps.TagID:
					return new IFDGps(aPefFile, tagID);
				case IFDJPEGInterchangeFormat.TagID:
					return new IFDJPEGInterchangeFormat(aPefFile, tagID);
				case IFDJPEGInterchangeFormatLength.TagID:
					return new IFDJPEGInterchangeFormatLength(aPefFile, tagID);
                //DNG:
                case IFDDNGVersion.TagID:
                    return new IFDDNGVersion(aPefFile, tagID);
                case IFDDNGBackwardVersion.TagID:
                    return new IFDDNGBackwardVersion(aPefFile, tagID);
                case IFDDNGBlackLevelRepeatDim.TagID:
                    return new IFDDNGBlackLevelRepeatDim(aPefFile, tagID);
                case IFDDNGCFALayout.TagID:
                    return new IFDDNGCFALayout(aPefFile, tagID);
                case IFDDNGCFAPlaneColor.TagID:
                    return new IFDDNGCFAPlaneColor(aPefFile, tagID);
                case IFDDNGLinearizationTable.TagID:
                    return new IFDDNGLinearizationTable(aPefFile, tagID);
                case IFDDNGLocalizedCameraModel.TagID:
                    return new IFDDNGLocalizedCameraModel(aPefFile, tagID);
                case IFDDNGUniqueCameraModel.TagID:
                    return new IFDDNGUniqueCameraModel(aPefFile, tagID);
                case IFDSubIFDs.TagID:
                    return new IFDSubIFDs(aPefFile, tagID);

                case IFDTileWidth.TagID:
                    return new IFDTileWidth(aPefFile, tagID);
                case IFDTileLength.TagID:
                    return new IFDTileLength(aPefFile, tagID);
                case IFDTileOffsets.TagID:
                    return new IFDTileOffsets(aPefFile, tagID);
                case IFDTileByteCounts.TagID:
                    return new IFDTileByteCounts(aPefFile, tagID);

                case IFDCFARepeatPatternDim.TagID:
                    return new IFDCFARepeatPatternDim(aPefFile, tagID);
                case IFDCFAPattern.TagID:
                    return new IFDCFAPattern(aPefFile, tagID);
                case IFDDNGAnalogBalance.TagID:

                    return new IFDDNGAnalogBalance(aPefFile, tagID);
                case IFDDNGAsShotNeutral.TagID:
                    return new IFDDNGAsShotNeutral(aPefFile, tagID);
                case IFDDNGBaselineExposure.TagID:
                    return new IFDDNGBaselineExposure(aPefFile, tagID);
                case IFDDNGBaselineNoise.TagID:
                    return new IFDDNGBaselineNoise(aPefFile, tagID);
                case IFDDNGBaselineSharpness.TagID:
                    return new IFDDNGBaselineSharpness(aPefFile, tagID);
                case IFDDNGCalibrationIlluminant1.TagID:
                    return new IFDDNGCalibrationIlluminant1(aPefFile, tagID);
                case IFDDNGCalibrationIlluminant2.TagID:
                    return new IFDDNGCalibrationIlluminant2(aPefFile, tagID);
                case IFDDNGColorMatrix1.TagID:
                    return new IFDDNGColorMatrix1(aPefFile, tagID);
                case IFDDNGColorMatrix2.TagID:
                    return new IFDDNGColorMatrix2(aPefFile, tagID);
                case IFDDNGForwardMatrix1.TagID:
                    return new IFDDNGForwardMatrix1(aPefFile, tagID);
                case IFDDNGForwardMatrix2.TagID:
                    return new IFDDNGForwardMatrix2(aPefFile, tagID);
                case IFDDNGImageNumber.TagID:
                    return new IFDDNGImageNumber(aPefFile, tagID);
                case IFDDNGLensInfo.TagID:
                    return new IFDDNGLensInfo(aPefFile, tagID);
                case IFDDNGLinearResponseLimit.TagID:
                    return new IFDDNGLinearResponseLimit(aPefFile, tagID);
                case IFDDNGOriginalRawFileName.TagID:
                    return new IFDDNGOriginalRawFileName(aPefFile, tagID);
                case IFDDNGPreviewApplicationName.TagID:
                    return new IFDDNGPreviewApplicationName(aPefFile, tagID);
                case IFDDNGPreviewApplicationVersion.TagID:
                    return new IFDDNGPreviewApplicationVersion(aPefFile, tagID);
                case IFDDNGPreviewColorSpace.TagID:
                    return new IFDDNGPreviewColorSpace(aPefFile, tagID);
                case IFDDNGPreviewDateTime.TagID:
                    return new IFDDNGPreviewDateTime(aPefFile, tagID);
                case IFDDNGPreviewSettingsDigest.TagID:
                    return new IFDDNGPreviewSettingsDigest(aPefFile, tagID);
                case IFDDNGPrivateData.TagID:
                    return new IFDDNGPrivateData(aPefFile, tagID);
                case IFDDNGProfileCalibrationSignature.TagID:
                    return new IFDDNGProfileCalibrationSignature(aPefFile, tagID);
                case IFDDNGProfileCopyright.TagID:
                    return new IFDDNGProfileCopyright(aPefFile, tagID);
                case IFDDNGProfileEmbedPolicy.TagID:
                    return new IFDDNGProfileEmbedPolicy(aPefFile, tagID);
                case IFDDNGProfileLookTableData.TagID:
                    return new IFDDNGProfileLookTableData(aPefFile, tagID);
                case IFDDNGProfileLookTableDims.TagID:
                    return new IFDDNGProfileLookTableDims(aPefFile, tagID);
                case IFDDNGProfileName.TagID:
                    return new IFDDNGProfileName(aPefFile, tagID);
                case IFDDNGRawDataUniqueID.TagID:
                    return new IFDDNGRawDataUniqueID(aPefFile, tagID);
                case IFDDNGRawImageDigest.TagID:
                    return new IFDDNGRawImageDigest(aPefFile, tagID);
                case IFDDNGShadowScale.TagID:
                    return new IFDDNGShadowScale(aPefFile, tagID);
                case IFDDNGTimeZoneOffset.TagID:
                    return new IFDDNGTimeZoneOffset(aPefFile, tagID);
                case IFDDNGXMPMetaData.TagID:
                    return new IFDDNGXMPMetaData(aPefFile, tagID);
                case IFDDNGBlackLevel.TagID:
                    return new IFDDNGBlackLevel(aPefFile, tagID);
                case IFDDNGBlackLevelDeltaH.TagID:
                    return new IFDDNGBlackLevelDeltaH(aPefFile, tagID);
                case IFDDNGBlackLevelDeltaV.TagID:
                    return new IFDDNGBlackLevelDeltaV(aPefFile, tagID);
                case IFDDNGWhiteLevel.TagID:
                    return new IFDDNGWhiteLevel(aPefFile, tagID);
                case IFDDNGDefaultScale.TagID:
                    return new IFDDNGDefaultScale(aPefFile, tagID);
                case IFDDNGDefaultCropOrigin.TagID:
                    return new IFDDNGDefaultCropOrigin(aPefFile, tagID);
                case IFDDNGDefaultCropSize.TagID:
                    return new IFDDNGDefaultCropSize(aPefFile, tagID);
                case IFDDNGBayerGreenSplit.TagID:
                    return new IFDDNGBayerGreenSplit(aPefFile, tagID);
                case IFDDNGChromaBlurRadius.TagID:
                    return new IFDDNGChromaBlurRadius(aPefFile, tagID);
                case IFDDNGActiveArea.TagID:
                    return new IFDDNGActiveArea(aPefFile, tagID);
                case IFDDNGBestQualityScale.TagID:
                    return new IFDDNGBestQualityScale(aPefFile, tagID);
                case IFDDNGAntiAliasStrength.TagID:
                    return new IFDDNGAntiAliasStrength(aPefFile, tagID);

                default:
					return new ImageFileDirectoryEntry(aPefFile, tagID);
			}
		}
	}
	#region Typed base classes
	public class StringImageFileDirectoryEntry : ImageFileDirectoryEntry
	{
		protected string mValue;

		public StringImageFileDirectoryEntry(FileReader aPefFile, ushort aTagID)
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

	public class ByteImageFileDirectoryEntry : ImageFileDirectoryEntry
	{
		protected byte[] mValue;

		public ByteImageFileDirectoryEntry(FileReader aPefFile, ushort aTagID)
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

	public class SByteImageFileDirectoryEntry : ImageFileDirectoryEntry
	{
		protected sbyte[] mValue;

		public SByteImageFileDirectoryEntry(FileReader aPefFile, ushort aTagID)
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

	public class UShortImageFileDirectoryEntry : ImageFileDirectoryEntry
	{
		protected ushort[] mValue;

		public UShortImageFileDirectoryEntry(FileReader aPefFile, ushort aTagID)
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

	public class ShortImageFileDirectoryEntry : ImageFileDirectoryEntry
	{
		protected short[] mValue;

		public ShortImageFileDirectoryEntry(FileReader aPefFile, ushort aTagID)
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

	public class IntImageFileDirectoryEntry : ImageFileDirectoryEntry
	{
		protected int[] mValue;

		public IntImageFileDirectoryEntry(FileReader aPefFile, ushort aTagID)
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

	public class UIntImageFileDirectoryEntry : ImageFileDirectoryEntry
	{
		protected uint[] mValue;

		public UIntImageFileDirectoryEntry(FileReader aPefFile, ushort aTagID)
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

	public class RationalImageFileDirectoryEntry : ImageFileDirectoryEntry
	{
		protected Rational[] mValue;

		public RationalImageFileDirectoryEntry(FileReader aPefFile, ushort aTagID)
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

	public class SRationalImageFileDirectoryEntry : ImageFileDirectoryEntry
	{
		protected SRational[] mValue;

		public SRationalImageFileDirectoryEntry(FileReader aPefFile, ushort aTagID)
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

	public class FloatImageFileDirectoryEntry : ImageFileDirectoryEntry
	{
		protected float[] mValue;

		public FloatImageFileDirectoryEntry(FileReader aPefFile, ushort aTagID)
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

	public class DoubleImageFileDirectoryEntry : ImageFileDirectoryEntry
	{
		protected double[] mValue;

		public DoubleImageFileDirectoryEntry(FileReader aPefFile, ushort aTagID)
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

	public class IFDArtist : StringImageFileDirectoryEntry
	{
		public IFDArtist(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 315;

		public const string TagName = "Artist";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue;
		}
	}

	public class IFDBitsPerSample : UShortImageFileDirectoryEntry
	{
		public IFDBitsPerSample(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }
		
		public const ushort TagID = 258;

		public const string TagName = "Bits per sample";

		public ushort Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class IFDCellLength : UShortImageFileDirectoryEntry
	{
		public IFDCellLength(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 265;

		public const string TagName = "Cell length";

		public ushort Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class IFDCellWidth : UShortImageFileDirectoryEntry
	{
		public IFDCellWidth(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 264;

		public const string TagName = "Cell width";

		public ushort Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class IFDColorMap : UShortImageFileDirectoryEntry
	{
		public IFDColorMap(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 320;

		public const string TagName = "Color map";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue.Length.ToString() + " entries.";
		}
	}

	public class IFDCompression : UShortImageFileDirectoryEntry
	{
		public enum Compression : ushort
		{ 
			NoCompression = 1,
			CCITTGroup3 = 2,
			PackBits = 32773,
			Pentax = 65535
		}

		public IFDCompression(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 259;

		public const string TagName = "Compression";

		public Compression Value
		{
			get { return (Compression)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class IFDCopyright : StringImageFileDirectoryEntry
	{
		public IFDCopyright(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 33432;

		public const string TagName = "Copyright";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue;
		}
	}

	public class IFDDateTime : StringImageFileDirectoryEntry
	{
		DateTime dt_value;

		public IFDDateTime(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			int year = int.Parse(mValue.Substring(0, 4));
			int month = int.Parse(mValue.Substring(5, 2));
			int day = int.Parse(mValue.Substring(8, 2));
			int hour = int.Parse(mValue.Substring(11, 2));
			int min = int.Parse(mValue.Substring(14, 2));
			int sec = int.Parse(mValue.Substring(17, 2));
			dt_value = new DateTime(year, month, day, hour, min, sec);
		}

		public const ushort TagID = 306;

		public const string TagName = "Date/Time";

		public DateTime Value
		{
			get { return dt_value; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class IFDExtraSamples : UShortImageFileDirectoryEntry
	{
		public IFDExtraSamples(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 338;

		public const string TagName = "Extra samples";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue.Length.ToString() + " entries.";
		}
	}

	public class IFDFillOrder : UShortImageFileDirectoryEntry
	{
		public IFDFillOrder(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 226;

		public const string TagName = "Fill order";

		public ushort Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue[0].ToString();
		}
	}

	public class IFDFreeByteCounts : UIntImageFileDirectoryEntry
	{
		public IFDFreeByteCounts(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 289;

		public const string TagName = "Free byte counts";

		public uint Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue[0].ToString();
		}
	}

	public class IFDFreeOffsets : UIntImageFileDirectoryEntry
	{
		public IFDFreeOffsets(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 288;

		public const string TagName = "Free offsets";

		public uint Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue[0].ToString();
		}
	}

	public class IFDGrayResponseCurve : UShortImageFileDirectoryEntry
	{
		public IFDGrayResponseCurve(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 291;

		public const string TagName = "Gray response curve";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue.Length.ToString() + " entries.";
		}
	}

	public class IFDGrayResponseUnit : UShortImageFileDirectoryEntry
	{
		public IFDGrayResponseUnit(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 290;

		public const string TagName = "Gray response unit";

		public uint Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue[0].ToString();
		}
	}

	public class IFDHostComputer : StringImageFileDirectoryEntry
	{
		public IFDHostComputer(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 316;

		public const string TagName = "Host computer";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue;
		}
	}

	public class IFDImageDescription : StringImageFileDirectoryEntry
	{
		public IFDImageDescription(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 270;

		public const string TagName = "Image description";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue;
		}
	}

	public class IFDImageLength : ImageFileDirectoryEntry
	{
		uint mValue;

		public IFDImageLength(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			if (mFieldType.SizeInBytes == 2 && aPefFile.EndianSwap)
			{
				unsafe
				{
					fixed (uint* ptr = &mOffset)
					{
						ushort* ptrUS = (ushort*)ptr;
						mValue = ptrUS[4 / mFieldType.SizeInBytes - 1];
					}
				}
			}
			else
				mValue = mOffset;
		}

		public const ushort TagID = 257;

		public const string TagName = "Image length";

		public uint Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue.ToString();
		}
	}

	public class IFDImageWidth : ImageFileDirectoryEntry
	{
		uint mValue;

		public IFDImageWidth(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			if (mFieldType.SizeInBytes == 2 && aPefFile.EndianSwap)
			{
				unsafe
				{
					fixed (uint* ptr = &mOffset)
					{
						ushort* ptrUS = (ushort*)ptr;
						mValue = ptrUS[4 / mFieldType.SizeInBytes - 1];
					}
				}
			}
			else
				mValue = mOffset;
		}

		public const ushort TagID = 256;

		public const string TagName = "Image width";

		public uint Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue.ToString();
		}
	}

	public class IFDMake : StringImageFileDirectoryEntry
	{
		public IFDMake(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 271;

		public const string TagName = "Make";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue;
		}
	}

	public class IFDMaxSampleValue : UShortImageFileDirectoryEntry
	{
		public IFDMaxSampleValue(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 281;

		public const string TagName = "Max sample value";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue.Length.ToString() + " entries.";
		}
	}

	public class IFDMinSampleValue : UShortImageFileDirectoryEntry
	{
		public IFDMinSampleValue(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 280;

		public const string TagName = "Min sample value";

		public ushort[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue.Length.ToString() + " entries.";
		}
	}

	public class IFDModel : StringImageFileDirectoryEntry
	{
		public IFDModel(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 272;

		public const string TagName = "Model";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue;
		}
	}

	public class IFDNewSubfileType : UIntImageFileDirectoryEntry
	{
		public IFDNewSubfileType(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 254;

		public const string TagName = "New subfile type";

		public uint Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue[0].ToString();
		}
	}

	public class IFDOrientation : UShortImageFileDirectoryEntry
	{
		public IFDOrientation(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 274;

		public const string TagName = "Orientation";

		public ushort Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue[0].ToString();
		}
	}

	public class IFDPhotometricInterpretation : UShortImageFileDirectoryEntry
	{
		public enum PhotometricInterpretation : ushort
		{ 
			WhiteIsZero = 0,
			BlackIsZero = 1,
			RGB = 2,
			Palette = 3,
			TransparencyMask = 4,
			PentaxRAW = 32803		
		}

		public IFDPhotometricInterpretation(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 262;

		public const string TagName = "Photometric interpretation";

		public PhotometricInterpretation Value
		{
			get { return (PhotometricInterpretation)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class IFDPlanarConfiguration : UShortImageFileDirectoryEntry
	{
		public enum PlanarConfigurartion : ushort
		{ 
			Chunky = 1,
			Planar = 2
		}

		public IFDPlanarConfiguration(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 284;

		public const string TagName = "Planar configuration";

		public PlanarConfigurartion Value
		{
			get { return (PlanarConfigurartion)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class IFDResolutionUnit : UShortImageFileDirectoryEntry
	{
		public enum ResolutionUnit
		{ 
			None = 1,
			Inch = 2,
			Centimeter = 3
		}

		public IFDResolutionUnit(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 296;

		public const string TagName = "Resolution unit";

		public ResolutionUnit Value
		{
			get { return (ResolutionUnit)mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.ToString();
		}
	}

	public class IFDRowsPerStrip : ImageFileDirectoryEntry
	{
		uint mValue;

		public IFDRowsPerStrip(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			if (mFieldType.SizeInBytes == 2 && aPefFile.EndianSwap)
			{
				unsafe
				{
					fixed (uint* ptr = &mOffset)
					{
						ushort* ptrUS = (ushort*)ptr;
						mValue = ptrUS[4 / mFieldType.SizeInBytes - 1];
					}
				}
			}
			else
				mValue = mOffset;
		}

		public const ushort TagID = 278;

		public const string TagName = "Rows per strip";

		public uint Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue.ToString();
		}
	}

	public class IFDSamplesPerPixel : UShortImageFileDirectoryEntry
	{
		public IFDSamplesPerPixel(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 277;

		public const string TagName = "Samples per pixel";

		public ushort Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue[0].ToString();
		}
	}

	public class IFDSoftware : StringImageFileDirectoryEntry
	{
		public IFDSoftware(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 305;

		public const string TagName = "Software";

		public string Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue;
		}
	}

	public class IFDStripByteCounts : ImageFileDirectoryEntry
	{
		uint[] mValue;

		public IFDStripByteCounts(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			if (mFieldType.SizeInBytes == 2)
			{
				if (mFieldType.SizeInBytes * mValueCount <= 4)
				{
					unsafe
					{
						fixed (uint* ptr = &mOffset)
						{
							ushort* ptrUS = (ushort*)ptr;
							mValue = new uint[mValueCount];
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

					mValue = new uint[mValueCount];
					for (int i = 0; i < mValueCount; i++)
					{
						mValue[i] = mPefFile.ReadUI2();
					}
					mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
				}
			}
			else
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

		public const ushort TagID = 279;

		public const string TagName = "Strip byte counts";

		public uint[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue.Length.ToString() + " entries.";
		}
	}

	public class IFDStripOffsets : ImageFileDirectoryEntry
	{
		uint[] mValue;

		public IFDStripOffsets(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			if (mFieldType.SizeInBytes == 2)
			{
				if (mFieldType.SizeInBytes * mValueCount <= 4)
				{
					unsafe
					{
						fixed (uint* ptr = &mOffset)
						{
							ushort* ptrUS = (ushort*)ptr;
							mValue = new uint[mValueCount];
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

					mValue = new uint[mValueCount];
					for (int i = 0; i < mValueCount; i++)
					{
						mValue[i] = mPefFile.ReadUI2();
					}
					mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
				}
			}
			else
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

		public const ushort TagID = 273;

		public const string TagName = "Strip offsets";

		public uint[] Value
		{
			get { return mValue; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue.Length.ToString() + " entries.";
		}
	}

	public class IFDSubfileType : UShortImageFileDirectoryEntry
	{
		public IFDSubfileType(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 255;

		public const string TagName = "Subfile type";

		public ushort Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue[0].ToString();
		}
	}

	public class IFDThreshholding : UShortImageFileDirectoryEntry
	{
		public IFDThreshholding(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 263;

		public const string TagName = "Threshholding";

		public ushort Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return mValue[0].ToString();
		}
	}

	public class IFDXResolution : RationalImageFileDirectoryEntry
	{
		public IFDXResolution(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 282;

		public const string TagName = "X-Resolution";

		public Rational Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue[0].ToString();
		}
	}

	public class IFDYResolution : RationalImageFileDirectoryEntry
	{
		public IFDYResolution(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 283;

		public const string TagName = "Y-Resolution";

		public Rational Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue[0].ToString();
		}
	}

	public class IFDExif : UIntImageFileDirectoryEntry
	{
		List<ExifEntry> mExif;

		public IFDExif(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			mExif = new List<ExifEntry>();
			uint currPos = mPefFile.Position();
			mPefFile.Seek(mValue[0], System.IO.SeekOrigin.Begin);

			ushort entryCount = mPefFile.ReadUI2();
			for (ushort i = 0; i < entryCount; i++)
			{
				ExifEntry entry = ExifEntry.CreateExifEntry(mPefFile);
				mExif.Add(entry);
			}
			mPefFile.Seek(currPos, System.IO.SeekOrigin.Begin);
		}

		public const ushort TagID = 34665;

		public const string TagName = "Exif";

		public List<ExifEntry> Value
		{
			get { return mExif; }
		}

		public T GetEntry<T>() where T : ExifEntry
		{
			Type t = typeof(T);
			foreach (var item in mExif)
			{
				if (item is T)
					return (T)item;
			}
			return null;
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Count.ToString() + " entries";
		}
	}

	public class IFDGps : UIntImageFileDirectoryEntry
	{
		List<GPSDirectoryEntry> mGPS;

		public IFDGps(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{
			mGPS = new List<GPSDirectoryEntry>();
			uint currPos = mPefFile.Position();
			mPefFile.Seek(mValue[0], System.IO.SeekOrigin.Begin);
			
			ushort entryCount = mPefFile.ReadUI2();
			for (ushort i = 0; i < entryCount; i++)
			{
				GPSDirectoryEntry entry = GPSDirectoryEntry.CreateGPSDirectoryEntry(mPefFile);
				mGPS.Add(entry);
			}
			mPefFile.Seek(currPos, System.IO.SeekOrigin.Begin);
		}

		public const ushort TagID = 34853;

		public const string TagName = "GPS";

		public List<GPSDirectoryEntry> Value
		{
			get { return mGPS; }
		}

		public override string ToString()
		{
			return TagName + ": " + Value.Count.ToString() + " entries";
		}
	}

	public class IFDJPEGInterchangeFormat : UIntImageFileDirectoryEntry
	{
		public IFDJPEGInterchangeFormat(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 513;

		public const string TagName = "JPEG Interchange format";

		public uint Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue[0].ToString();
		}
	}

	public class IFDJPEGInterchangeFormatLength : UIntImageFileDirectoryEntry
	{
		public IFDJPEGInterchangeFormatLength(FileReader aPefFile, ushort aTagID)
			: base(aPefFile, aTagID)
		{ }

		public const ushort TagID = 514;

		public const string TagName = "JPEG Interchange format length";

		public uint Value
		{
			get { return mValue[0]; }
		}

		public override string ToString()
		{
			return TagName + ": " + mValue[0].ToString();
		}
	}

    #region DNG Tags
    public class IFDDNGVersion : ByteImageFileDirectoryEntry
    {
        public IFDDNGVersion(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50706;

        public const string TagName = "DNGVersion";

        public byte[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + Value[0].ToString() + Value[1].ToString() + Value[2].ToString() + Value[3].ToString();
        }
    }

    public class IFDDNGBackwardVersion : ByteImageFileDirectoryEntry
    {
        public IFDDNGBackwardVersion(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50707;

        public const string TagName = "DNGBackwardVersion";

        public byte[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + Value[0].ToString() + Value[1].ToString() + Value[2].ToString() + Value[3].ToString();
        }
    }

    public class IFDDNGUniqueCameraModel : StringImageFileDirectoryEntry
    {
        public IFDDNGUniqueCameraModel(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50708;

        public const string TagName = "Unique Camera Model";

        public string Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue;
        }
    }

    public class IFDDNGLocalizedCameraModel : StringImageFileDirectoryEntry
    {
        public IFDDNGLocalizedCameraModel(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50709;

        public const string TagName = "Localized Camera Model";

        public string Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue;
        }
    }

    public class IFDDNGCFAPlaneColor : ByteImageFileDirectoryEntry
    {
        public IFDDNGCFAPlaneColor(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50710;

        public const string TagName = "CFA Plane Color";

        public byte[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            string ret = TagName + ": ";
            for (int i = 0; i < mValue.Length; i++)
            {
                ret += Value[i].ToString();
            }

            return ret;
        }
    }

    public class IFDDNGCFALayout : ShortImageFileDirectoryEntry
    {
        public IFDDNGCFALayout(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50711;

        public const string TagName = "CFA Layout";

        public short[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDDNGLinearizationTable : UShortImageFileDirectoryEntry
    {
        public IFDDNGLinearizationTable(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50712;

        public const string TagName = "Linearization Table";

        public ushort[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.Length.ToString() + " entries.";
        }
    }

    public class IFDDNGBlackLevelRepeatDim : ShortImageFileDirectoryEntry
    {
        public IFDDNGBlackLevelRepeatDim(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50713;

        public const string TagName = "Black Level Repeat Dim";

        public short[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString() + " " + mValue[1].ToString();
        }
    }

    public class IFDTileWidth : ImageFileDirectoryEntry
    {
        uint[] mValue;

        public IFDTileWidth(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        {
            if (mFieldType.SizeInBytes == 2)
            {
                if (mFieldType.SizeInBytes * mValueCount <= 4)
                {
                    unsafe
                    {
                        fixed (uint* ptr = &mOffset)
                        {
                            ushort* ptrUS = (ushort*)ptr;
                            mValue = new uint[mValueCount];
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

                    mValue = new uint[mValueCount];
                    for (int i = 0; i < mValueCount; i++)
                    {
                        mValue[i] = mPefFile.ReadUI2();
                    }
                    mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
                }
            }
            else
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

        public const ushort TagID = 322;

        public const string TagName = "Tile Width";

        public uint Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDTileLength : ImageFileDirectoryEntry
    {
        uint[] mValue;

        public IFDTileLength(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        {
            if (mFieldType.SizeInBytes == 2)
            {
                if (mFieldType.SizeInBytes * mValueCount <= 4)
                {
                    unsafe
                    {
                        fixed (uint* ptr = &mOffset)
                        {
                            ushort* ptrUS = (ushort*)ptr;
                            mValue = new uint[mValueCount];
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

                    mValue = new uint[mValueCount];
                    for (int i = 0; i < mValueCount; i++)
                    {
                        mValue[i] = mPefFile.ReadUI2();
                    }
                    mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
                }
            }
            else
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

        public const ushort TagID = 323;

        public const string TagName = "Tile Length";

        public uint Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDTileByteCounts : ImageFileDirectoryEntry
    {
        uint[] mValue;

        public IFDTileByteCounts(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        {
            if (mFieldType.SizeInBytes == 2)
            {
                if (mFieldType.SizeInBytes * mValueCount <= 4)
                {
                    unsafe
                    {
                        fixed (uint* ptr = &mOffset)
                        {
                            ushort* ptrUS = (ushort*)ptr;
                            mValue = new uint[mValueCount];
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

                    mValue = new uint[mValueCount];
                    for (int i = 0; i < mValueCount; i++)
                    {
                        mValue[i] = mPefFile.ReadUI2();
                    }
                    mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
                }
            }
            else
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

        public const ushort TagID = 325;

        public const string TagName = "Tile Byte Counts";

        public uint[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString() + " + " + (mValue.Length - 1).ToString() + " more...";
        }
    }

    public class IFDTileOffsets : UIntImageFileDirectoryEntry
    {
        public IFDTileOffsets(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 324;

        public const string TagName = "Tile Offsets";

        public uint[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString() + " + " + (mValue.Length - 1).ToString() + " more...";
        }
    }

    public class IFDSubIFDs : ImageFileDirectoryEntry
    {
        List<ImageFileDirectory> mValue;

        public IFDSubIFDs(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        {
            uint[] offsets = new uint[mValueCount];
            uint currentOffset;

            if (mFieldType.SizeInBytes == 2)
            {
                if (mFieldType.SizeInBytes * mValueCount <= 4)
                {
                    unsafe
                    {
                        fixed (uint* ptr = &mOffset)
                        {
                            ushort* ptrUS = (ushort*)ptr;
                            offsets = new uint[mValueCount];
                            for (int i = 0; i < mValueCount; i++)
                            {
                                if (aPefFile.EndianSwap)
                                    offsets[i] = ptrUS[4 / mFieldType.SizeInBytes - i - 1];
                                else
                                    offsets[i] = ptrUS[i];
                            }
                        }
                    }
                }
                else
                {
                    currentOffset = mPefFile.Position();
                    mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

                    offsets = new uint[mValueCount];
                    for (int i = 0; i < mValueCount; i++)
                    {
                        offsets[i] = mPefFile.ReadUI2();
                    }
                    mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
                }
            }
            else
            {
                if (mFieldType.SizeInBytes * mValueCount <= 4)
                {
                    offsets = new uint[mValueCount];
                    offsets[0] = mOffset;
                }
                else
                {
                    currentOffset = mPefFile.Position();
                    mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

                    offsets = new uint[mValueCount];
                    for (int i = 0; i < mValueCount; i++)
                    {
                        offsets[i] = mPefFile.ReadUI4();
                    }
                    mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
                }
            }


            currentOffset = mPefFile.Position();

            mValue = new List<ImageFileDirectory>();

            foreach (var item in offsets)
            {
                mPefFile.Seek(item, System.IO.SeekOrigin.Begin);
                ImageFileDirectory ifd = new ImageFileDirectory(aPefFile);
                mValue.Add(ifd);
            }

            mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);

        }

        public const ushort TagID = 330;

        public const string TagName = "SubIFDs";

        public List<ImageFileDirectory> Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.ToString();
        }
    }

    public class IFDCFARepeatPatternDim : UShortImageFileDirectoryEntry
    {
        public IFDCFARepeatPatternDim(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 33421;

        public const string TagName = "CFA Repeat Pattern Dim";

        public ushort[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString() + ", " + mValue[1].ToString();
        }
    }

    public class IFDCFAPattern : ByteImageFileDirectoryEntry
    {
        public IFDCFAPattern(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 33422;

        public const string TagName = "CFA Pattern";

        public ExifEntry.ExifCFAPattern.BayerColor[] Value
        {
            get
            {
                ExifEntry.ExifCFAPattern.BayerColor[] ret = new ExifEntry.ExifCFAPattern.BayerColor[mValue.Length];
                for (int i = 0; i < mValue.Length; i++)
                {
                    ret[i] = (ExifEntry.ExifCFAPattern.BayerColor)mValue[i];
                }
                return ret;
            }
        }

        public override string ToString()
        {
            string ret = TagName + ": ";

            for (int i = 0; i < mValue.Length; i++)
            {
                ret += mValue[i];
                if (i < mValue.Length - 1)
                    ret += ", ";
            }

            return ret;
        }
    }

    public class IFDDNGTimeZoneOffset : ShortImageFileDirectoryEntry
    {
        public IFDDNGTimeZoneOffset(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 34858;

        public const string TagName = "Time Zone Offset";

        public short Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDDNGImageNumber : IntImageFileDirectoryEntry
    {
        public IFDDNGImageNumber(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 37393;

        public const string TagName = "Image Number";

        public int Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDDNGColorMatrix1 : SRationalImageFileDirectoryEntry
    {
        public IFDDNGColorMatrix1(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50721;

        public const string TagName = "Color Matrix 1";

        public SRational[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            string ret = TagName + "{{";
            for (int y = 0; y < 3; y++)
            {
                for (int x = 0; x < 3; x++)
                {
                    ret += mValue[y * 3 + x].ToString();
                    if (x < 2) ret += "; ";
                }
                if (y < 2) ret += "}, {";
            }
            ret += "}}";
            return ret;
        }
    }

    public class IFDDNGColorMatrix2 : SRationalImageFileDirectoryEntry
    {
        public IFDDNGColorMatrix2(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50722;

        public const string TagName = "Color Matrix 2";

        public SRational[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            string ret = TagName + "{{";
            for (int y = 0; y < 3; y++)
            {
                for (int x = 0; x < 3; x++)
                {
                    ret += mValue[y * 3 + x].ToString();
                    if (x < 2) ret += "; ";
                }
                if (y < 2) ret += "}, {";
            }
            ret += "}}";
            return ret;
        }
    }

    public class IFDDNGAnalogBalance : RationalImageFileDirectoryEntry
    {
        public IFDDNGAnalogBalance(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50727;

        public const string TagName = "Analog Balance";

        public Rational[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.Length.ToString() + " entries";
        }
    }

    public class IFDDNGAsShotNeutral : ImageFileDirectoryEntry
    {
        Rational[] mValue;

        public IFDDNGAsShotNeutral(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        {
            if (mFieldType.SizeInBytes == 2)
            {
                if (mFieldType.SizeInBytes * mValueCount <= 4)
                {
                    unsafe
                    {
                        fixed (uint* ptr = &mOffset)
                        {
                            ushort* ptrUS = (ushort*)ptr;
                            mValue = new Rational[mValueCount];
                            for (int i = 0; i < mValueCount; i++)
                            {
                                if (aPefFile.EndianSwap)
                                    mValue[i] = new Rational(ptrUS[4 / mFieldType.SizeInBytes - i - 1], 1);
                                else
                                    mValue[i] = new Rational(ptrUS[i], 1);
                            }
                        }
                    }
                }
                else
                {
                    uint currentOffset = mPefFile.Position();
                    mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

                    mValue = new Rational[mValueCount];
                    for (int i = 0; i < mValueCount; i++)
                    {
                        mValue[i] = new Rational(mPefFile.ReadUI2(),1);
                    }
                    mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
                }
            }
            else
            {
                uint currentOffset = mPefFile.Position();
                mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

                mValue = new Rational[mValueCount];
                for (int i = 0; i < mValueCount; i++)
                {
                    mValue[i].numerator = mPefFile.ReadUI4();
                    mValue[i].denominator = mPefFile.ReadUI4();
                }
                mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
            }
        }

        public const ushort TagID = 50728;

        public const string TagName = "As Shot Neutral";

        public Rational[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString() + " + " + (mValue.Length - 1).ToString() + " more...";
        }
    }

    public class IFDDNGBaselineExposure : SRationalImageFileDirectoryEntry
    {
        public IFDDNGBaselineExposure(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50730;

        public const string TagName = "Baseline Exposure";

        public SRational Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString() + " ev";
        }
    }

    public class IFDDNGBaselineNoise : RationalImageFileDirectoryEntry
    {
        public IFDDNGBaselineNoise(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50731;

        public const string TagName = "Baseline Noise";

        public Rational Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDDNGBaselineSharpness : RationalImageFileDirectoryEntry
    {
        public IFDDNGBaselineSharpness(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50732;

        public const string TagName = "Baseline Sharpness";

        public Rational Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDDNGLinearResponseLimit : RationalImageFileDirectoryEntry
    {
        public IFDDNGLinearResponseLimit(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50734;

        public const string TagName = "Linear Response Limit";

        public Rational Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDDNGLensInfo : RationalImageFileDirectoryEntry
    {
        public IFDDNGLensInfo(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50736;

        public const string TagName = "Lens Info";

        public Rational[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString() + " - " + mValue[1].ToString() +
                ", " + mValue[2].ToString() +" - " + mValue[3].ToString();
        }
    }

    public class IFDDNGShadowScale : RationalImageFileDirectoryEntry
    {
        public IFDDNGShadowScale(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50739;

        public const string TagName = "Shadow Scale";

        public Rational Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDDNGPrivateData : ByteImageFileDirectoryEntry
    {
        public IFDDNGPrivateData(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50740;

        public const string TagName = "DNG Private Data";

        public byte[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.Length.ToString();
        }
    }

    public class IFDDNGCalibrationIlluminant1 : UShortImageFileDirectoryEntry
    {
        public IFDDNGCalibrationIlluminant1(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50778;

        public const string TagName = "Calibration Illuminant 1";

        public ushort Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDDNGCalibrationIlluminant2 : UShortImageFileDirectoryEntry
    {
        public IFDDNGCalibrationIlluminant2(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50779;

        public const string TagName = "Calibration Illuminant 2";

        public ushort Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDDNGRawDataUniqueID : ByteImageFileDirectoryEntry
    {
        public IFDDNGRawDataUniqueID(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50781;

        public const string TagName = "Raw Data Unique ID";

        public byte[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.Length.ToString() + " entries...";
        }
    }

    public class IFDDNGOriginalRawFileName : StringImageFileDirectoryEntry
    {
        public IFDDNGOriginalRawFileName(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50827;

        public const string TagName = "Original Raw File Name";

        public string Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue;
        }
    }

    public class IFDDNGProfileCalibrationSignature : StringImageFileDirectoryEntry
    {
        public IFDDNGProfileCalibrationSignature(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50932;

        public const string TagName = "Profile Calibration Signature";

        public string Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue;
        }
    }

    public class IFDDNGProfileName : StringImageFileDirectoryEntry
    {
        public IFDDNGProfileName(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50936;

        public const string TagName = "Profile Name";

        public string Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue;
        }
    }

    public class IFDDNGProfileEmbedPolicy : UIntImageFileDirectoryEntry
    {
        public IFDDNGProfileEmbedPolicy(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50941;

        public const string TagName = "Profile Embed Policy";

        public uint Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDDNGProfileCopyright : StringImageFileDirectoryEntry
    {
        public IFDDNGProfileCopyright(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50942;

        public const string TagName = "Profile Copyright";

        public string Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue;
        }
    }

    public class IFDDNGForwardMatrix1 : SRationalImageFileDirectoryEntry
    {
        public IFDDNGForwardMatrix1(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50964;

        public const string TagName = "Forward Matrix 1";

        public SRational[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.Length.ToString() + " entries...";
        }
    }

    public class IFDDNGForwardMatrix2 : SRationalImageFileDirectoryEntry
    {
        public IFDDNGForwardMatrix2(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50965;

        public const string TagName = "Forward Matrix 2";

        public SRational[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.Length.ToString() + " entries...";
        }
    }

    public class IFDDNGPreviewApplicationName : StringImageFileDirectoryEntry
    {
        public IFDDNGPreviewApplicationName(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50966;

        public const string TagName = "Preview Application Name";

        public string Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue;
        }
    }

    public class IFDDNGPreviewApplicationVersion : StringImageFileDirectoryEntry
    {
        public IFDDNGPreviewApplicationVersion(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50967;

        public const string TagName = "Preview Application Version";

        public string Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue;
        }
    }

    public class IFDDNGPreviewSettingsDigest : ByteImageFileDirectoryEntry
    {
        public IFDDNGPreviewSettingsDigest(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50969;

        public const string TagName = "Preview Settings Digest";

        public byte[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.Length.ToString() + " entries...";
        }
    }

    public class IFDDNGPreviewColorSpace : UIntImageFileDirectoryEntry
    {
        public IFDDNGPreviewColorSpace(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50970;

        public const string TagName = "Preview Color Space";

        public uint Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDDNGPreviewDateTime : StringImageFileDirectoryEntry
    {
        public IFDDNGPreviewDateTime(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50971;

        public const string TagName = "Preview Date Time";

        public string Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue;
        }
    }

    public class IFDDNGRawImageDigest : ByteImageFileDirectoryEntry
    {
        public IFDDNGRawImageDigest(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50972;

        public const string TagName = "RawImageDigest";

        public byte[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.Length.ToString() + " entries...";
        }
    }

    public class IFDDNGProfileLookTableDims : UIntImageFileDirectoryEntry
    {
        public IFDDNGProfileLookTableDims(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50981;

        public const string TagName = "Profile Look Table Dims";

        public uint[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString() + ", " + mValue[2].ToString() + ", " + mValue[3].ToString();
        }
    }

    public class IFDDNGProfileLookTableData : FloatImageFileDirectoryEntry
    {
        public IFDDNGProfileLookTableData(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50982;

        public const string TagName = "Profile Look Table Dims";

        public float[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.Length.ToString() + " entries...";
        }
    }

    public class IFDDNGXMPMetaData : StringImageFileDirectoryEntry
    {
        public IFDDNGXMPMetaData(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 700;

        public const string TagName = "XMP Meta Data";

        public string Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue;
        }
    }

    public class IFDDNGBlackLevel : ImageFileDirectoryEntry
    {
        Rational[] mValue;

        public IFDDNGBlackLevel(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        {
            if (mFieldType.SizeInBytes == 2)
            {
                if (mFieldType.SizeInBytes * mValueCount <= 4)
                {
                    unsafe
                    {
                        fixed (uint* ptr = &mOffset)
                        {
                            ushort* ptrUS = (ushort*)ptr;
                            mValue = new Rational[mValueCount];
                            for (int i = 0; i < mValueCount; i++)
                            {
                                if (aPefFile.EndianSwap)
                                    mValue[i] = new Rational(ptrUS[4 / mFieldType.SizeInBytes - i - 1], 1);
                                else
                                    mValue[i] = new Rational(ptrUS[i], 1);
                            }
                        }
                    }
                }
                else
                {
                    uint currentOffset = mPefFile.Position();
                    mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

                    mValue = new Rational[mValueCount];
                    for (int i = 0; i < mValueCount; i++)
                    {
                        mValue[i] = new Rational(mPefFile.ReadUI2(), 1);
                    }
                    mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
                }
            }
            else
            if (mFieldType.SizeInBytes == 4)
            {
                if (mFieldType.SizeInBytes * mValueCount <= 4)
                {
                    unsafe
                    {
                        fixed (uint* ptr = &mOffset)
                        {
                            mValue = new Rational[mValueCount];
                            for (int i = 0; i < mValueCount; i++)
                            {
                                mValue[i] = new Rational(ptr[i], 1);
                            }
                        }
                    }
                }
                else
                {
                    uint currentOffset = mPefFile.Position();
                    mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

                    mValue = new Rational[mValueCount];
                    for (int i = 0; i < mValueCount; i++)
                    {
                        mValue[i] = new Rational(mPefFile.ReadUI4(), 1);
                    }
                    mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
                }
            }
            else
            {
                uint currentOffset = mPefFile.Position();
                mPefFile.Seek(mOffset, System.IO.SeekOrigin.Begin);

                mValue = new Rational[mValueCount];
                for (int i = 0; i < mValueCount; i++)
                {
                    mValue[i].numerator = mPefFile.ReadUI4();
                    mValue[i].denominator = mPefFile.ReadUI4();
                }
                mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
            }
        }

        public const ushort TagID = 50714;

        public const string TagName = "Black Level";

        public Rational[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString() + " + " + (mValue.Length - 1).ToString() + " more...";
        }
    }

    public class IFDDNGBlackLevelDeltaH : SRationalImageFileDirectoryEntry
    {
        public IFDDNGBlackLevelDeltaH(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50715;

        public const string TagName = "Black Level Delta H";

        public SRational[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.Length.ToString() + " entries...";
        }
    }

    public class IFDDNGBlackLevelDeltaV : SRationalImageFileDirectoryEntry
    {
        public IFDDNGBlackLevelDeltaV(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50716;

        public const string TagName = "Black Level Delta V";

        public SRational[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue.Length.ToString() + " entries...";
        }
    }

    public class IFDDNGWhiteLevel : ImageFileDirectoryEntry
    {
        uint[] mValue;

        public IFDDNGWhiteLevel(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        {
            if (mFieldType.SizeInBytes == 2)
            {
                if (mFieldType.SizeInBytes * mValueCount <= 4)
                {
                    unsafe
                    {
                        fixed (uint* ptr = &mOffset)
                        {
                            ushort* ptrUS = (ushort*)ptr;
                            mValue = new uint[mValueCount];
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

                    mValue = new uint[mValueCount];
                    for (int i = 0; i < mValueCount; i++)
                    {
                        mValue[i] = mPefFile.ReadUI2();
                    }
                    mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
                }
            }
            else
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

        public const ushort TagID = 50717;

        public const string TagName = "White Level";

        public uint[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString() + " + " + (mValue.Length - 1).ToString() + " more...";
        }
    }

    public class IFDDNGDefaultScale : RationalImageFileDirectoryEntry
    {
        public IFDDNGDefaultScale(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50718;

        public const string TagName = "Default Scale";

        public Rational[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString() + ", " + mValue[1].ToString();
        }
    }

    public class IFDDNGDefaultCropOrigin : UShortImageFileDirectoryEntry
    {
        public IFDDNGDefaultCropOrigin(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50719;

        public const string TagName = "Default Crop Origin";

        public ushort[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString() + " + " + (mValue.Length - 1).ToString() + " more...";
        }
    }

    public class IFDDNGDefaultCropSize : UShortImageFileDirectoryEntry
    {
        public IFDDNGDefaultCropSize(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50720;

        public const string TagName = "Default Crop Size";

        public ushort[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString() + " + " + (mValue.Length - 1).ToString() + " more...";
        }
    }

    public class IFDDNGBayerGreenSplit : UIntImageFileDirectoryEntry
    {
        public IFDDNGBayerGreenSplit(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50733;

        public const string TagName = "Bayer Green Split";

        public uint Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDDNGChromaBlurRadius : RationalImageFileDirectoryEntry
    {
        public IFDDNGChromaBlurRadius(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50737;

        public const string TagName = "Chroma Blur Radius";

        public Rational Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDDNGAntiAliasStrength : RationalImageFileDirectoryEntry
    {
        public IFDDNGAntiAliasStrength(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50738;

        public const string TagName = "Anti Alias Strength";

        public Rational Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }

    public class IFDDNGActiveArea : ImageFileDirectoryEntry
    {
        uint[] mValue;

        public IFDDNGActiveArea(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        {
            if (mFieldType.SizeInBytes == 2)
            {
                if (mFieldType.SizeInBytes * mValueCount <= 4)
                {
                    unsafe
                    {
                        fixed (uint* ptr = &mOffset)
                        {
                            ushort* ptrUS = (ushort*)ptr;
                            mValue = new uint[mValueCount];
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

                    mValue = new uint[mValueCount];
                    for (int i = 0; i < mValueCount; i++)
                    {
                        mValue[i] = mPefFile.ReadUI2();
                    }
                    mPefFile.Seek(currentOffset, System.IO.SeekOrigin.Begin);
                }
            }
            else
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

        public const ushort TagID = 50829;

        public const string TagName = "Active Area";

        public uint[] Value
        {
            get { return mValue; }
        }

        public override string ToString()
        {
            return TagName + ": (" + mValue[0].ToString() + ", " + mValue[1].ToString() + ") - (" + mValue[2].ToString() + ", " + mValue[3].ToString() + ")";
        }
    }

    public class IFDDNGBestQualityScale : RationalImageFileDirectoryEntry
    {
        public IFDDNGBestQualityScale(FileReader aPefFile, ushort aTagID)
            : base(aPefFile, aTagID)
        { }

        public const ushort TagID = 50780;

        public const string TagName = "Best Quality Scale";

        public Rational Value
        {
            get { return mValue[0]; }
        }

        public override string ToString()
        {
            return TagName + ": " + mValue[0].ToString();
        }
    }
    #endregion
}
