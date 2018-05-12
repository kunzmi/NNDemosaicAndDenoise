using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PentaxPefFile
{
	public class ImageFileDirectory
	{
		private FileReader mPefFile;
		private ushort mEntryCount;
		private List<ImageFileDirectoryEntry> mEntries;

		public ImageFileDirectory(FileReader aPefFile)
		{
			mPefFile = aPefFile;
			mEntries = new List<ImageFileDirectoryEntry>();
			mEntryCount = mPefFile.ReadUI2();
			for (ushort i = 0; i < mEntryCount; i++)
			{
				ImageFileDirectoryEntry entry = ImageFileDirectoryEntry.CreateImageFileDirectoryEntry(mPefFile);
				mEntries.Add(entry);
			}
		}

		public T GetEntry<T>() where T : ImageFileDirectoryEntry
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
