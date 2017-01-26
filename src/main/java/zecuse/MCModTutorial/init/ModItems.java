package zecuse.MCModTutorial.init;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import net.minecraftforge.fml.common.registry.GameRegistry;
import zecuse.MCModTutorial.item.GenericItem;
import zecuse.MCModTutorial.item.Leaf;
import zecuse.MCModTutorial.reference.Reference;

// This class loads the items into the game.
@GameRegistry.ObjectHolder(Reference.MOD_ID)
public class ModItems
{
	// This variable holds all of the mod item instances.
	// A List collection is used because the mod doesn't have too many items.
	// A Set collection should be used in that event.
	private static final List<GenericItem> ITEMS = new ArrayList<>();
	
	// List all of the mod item instances.
	public static final GenericItem LEAF = new Leaf();
	
	public static Collection<GenericItem> getItems()
	{
		return ITEMS;
	}
	
	// Add the mod item to the collection for later registration/usage.
	public static void register(GenericItem item)
	{
		ITEMS.add(item);
	}
}
