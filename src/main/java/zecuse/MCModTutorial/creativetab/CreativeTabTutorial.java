package zecuse.MCModTutorial.creativetab;

import net.minecraft.creativetab.CreativeTabs;
import net.minecraft.item.Item;
import zecuse.MCModTutorial.init.ModItems;
import zecuse.MCModTutorial.reference.Reference;

// This class creates a creative mode tab to store all of the items and blocks in the mod.
public class CreativeTabTutorial
{
	// Creates the tab with the mod id as its name.
	public static final CreativeTabs TUTORIAL_TAB = new CreativeTabs(Reference.MOD_ID)
	{
		// Gives the tab a custom icon.
		@Override
		public Item getTabIconItem()
		{
			return ModItems.LEAF;
		}
	};
}
