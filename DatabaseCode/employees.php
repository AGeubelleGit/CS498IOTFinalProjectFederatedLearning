<?php

echo "Employees Seen:";

$link = mysqli_connect("127.0.0.1", "user", "password", "final_project.db");
if ($link) {
	#echo "Successfully connected to database <br />";
	$query = mysqli_query($link, "SELECT * FROM employee_data ORDER BY timestamp DESC");
	
	echo "<table border=1>";
	echo "<tr><th>Employee Name</th><th>Employee ID</th><th>Timestamp</th> </tr>";
	while($array=mysqli_fetch_array($query)) {
		echo "<tr>";
		echo "<td>";
		echo $array['employee_name'];
        echo "</td>";
        
		echo "<td>";
		echo $array['employee_id'];
        echo "</td>";
        
		echo "<td>";
		echo $array['time'];
		echo "</td>";
		echo "</tr>";
	}
	echo "</table>";
}
else {
	echo "Failed to connect <br/>";
	echo "MySQL error : ".mysqli_error();
}
?>