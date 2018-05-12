using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PentaxPefFile
{
	public class PentaxMakerNotes : FileReader
	{
		List<PentaxMakerNotesEntry> mEntries;
		ushort mEntryCount;
		public bool K3Specific = false;

		public PentaxMakerNotes(byte[] aData)
			: base(new MemoryStream(aData))
		{
			string testVersion = ReadStr(4);
			if (testVersion == "AOC\0")
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
						throw new FileLoadException("Could not determine file endianess for maker notes");

				mEndianSwap = fileIsLittleEndian != BitConverter.IsLittleEndian;
			}
			else
			{
				Seek(0, SeekOrigin.Begin);
				testVersion = ReadStr(8);
				if (testVersion == "PENTAX \0")
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
							throw new FileLoadException("Could not determine file endianess for maker notes");

					mEndianSwap = fileIsLittleEndian != BitConverter.IsLittleEndian;
				}
			}

			mEntries = new List<PentaxMakerNotesEntry>();
			mEntryCount = ReadUI2();
			for (ushort i = 0; i < mEntryCount; i++)
			{
				PentaxMakerNotesEntry entry = PentaxMakerNotesEntry.CreatePentaxMakerNotesEntry(this);
				mEntries.Add(entry);
			}

			MNPreviewImageSize imagesize = GetEntry<MNPreviewImageSize>();
			MNPreviewImageLength imagelength = GetEntry<MNPreviewImageLength>();
			MNPreviewImageStart imagestart = GetEntry<MNPreviewImageStart>();
			MNPreviewImageBorders imageborder = GetEntry<MNPreviewImageBorders>();

			uint curPos = Position();
			Seek(imagestart.Value, SeekOrigin.Begin);
			byte[] data = mFileReader.ReadBytes((int)imagelength.Value);
			Seek(curPos, SeekOrigin.Begin);

		}

		public T GetEntry<T>() where T : PentaxMakerNotesEntry
		{
			Type t = typeof(T);
			foreach (var item in mEntries)
			{
				if (item is T)
					return (T)item;
			}
			return null;
		}
	}
}
