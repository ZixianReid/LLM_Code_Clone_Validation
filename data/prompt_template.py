from string import Template

task = "Assess whether the provided code snippets are code clones."

instruction_one = Template(
    """
1. Review one example to understand the task. The example starts with ##Example and ends with 
--.Each example includes two code snippets and a flag: '$flag_true' if the snippets are code clones. '$flag_false' if they are not.
2.After the example, you will be given a new pair of code snippets to assess.
"""
)

instruction_few = Template(
    """1. Review three examples to understand the task. The example starts with ##Example and ends with 
--.Each example includes two code snippets and a flag: '$flag_true' if the snippets are code clones. '$flag_false' if they are not. 
2.After the examples, you will be given a new pair of code snippets to assess."""
)

definition = ("Code clones are segments of code that are either identical, syntactically similar with minor "
              "variations, structurally similar but significantly modified, or perform the same functionality with "
              "different syntax.")

output_requirement = Template(
    "Output only '$flag_true' if the snippets are code clones, and only '$flag_false' if they are not. Provide no other "
    "output.")

example_few_bcb = Template(
    """
## Example
Code snippet 1:
    private static String encode(final String input) throws UnsupportedEncodingException, NoSuchAlgorithmException {
        MessageDigest md = MessageDigest.getInstance("MD5");
        md.reset();
        md.update(input.getBytes("UTF-8"));
        return toHexString(md.digest());
    }
Code snippet 2:
    public static byte[] encrypt(String x) throws Exception {
        java.security.MessageDigest d = null;
        d = java.security.MessageDigest.getInstance("SHA-1");
        d.reset();
        d.update(x.getBytes());
        return d.digest();
    }
Flag: $flag_true
--
## Example
Code snippet 1:
    HttpRepository(Path path) throws IOException {
        super(path);
        this.url = new URL(path.toURLString());
        HttpURLConnection.setFollowRedirects(true);
        this.connection = (HttpURLConnection) url.openConnection();
        this.ns = Names.getNamespace(path);
    }
Code snippet 2:     
    public static byte[] encrypt(String x) throws Exception {
        java.security.MessageDigest d = null;
        d = java.security.MessageDigest.getInstance("SHA-1");
        d.reset();
        d.update(x.getBytes());
        return d.digest();
    }
Flag: $flag_false
--
## Example
Code snippet 1: 
    public static byte[] createPasswordDigest(String password, byte[] salt) throws Exception {
        MessageDigest md = MessageDigest.getInstance("MD5");
        md.update(salt);
        md.update(password.getBytes("UTF8"));
        byte[] digest = md.digest();
        return digest;
    }
Code snippet 2:
    public static byte[] encrypt(String x) throws Exception {
        java.security.MessageDigest d = null;
        d = java.security.MessageDigest.getInstance("SHA-1");
        d.reset();
        d.update(x.getBytes());
        return d.digest();
    }
Flag: $flag_true
--
## Example
Code snippet 1:
    public static byte[] encrypt(String x) throws Exception {
        java.security.MessageDigest d = null;
        d = java.security.MessageDigest.getInstance("SHA-1");
        d.reset();
        d.update(x.getBytes());
        return d.digest();
    }
Code snippet 2:
    private static void copyFile(File src, File dst) throws IOException {
        FileChannel in = new FileInputStream(src).getChannel();
        FileChannel out = new FileOutputStream(dst).getChannel();
        in.transferTo(0, in.size(), out);
        in.close();
        out.close();
    }
Flag: $flag_false
--
"""
)

example_one_bcb = Template(
    """
## Example
Code snippet 1:
    public static byte[] encrypt(String x) throws Exception {
        java.security.MessageDigest d = null;
        d = java.security.MessageDigest.getInstance("SHA-1");
        d.reset();
        d.update(x.getBytes());
        return d.digest();
    }
Code snippet 2:
    private static void copyFile(File src, File dst) throws IOException {
        FileChannel in = new FileInputStream(src).getChannel();
        FileChannel out = new FileOutputStream(dst).getChannel();
        in.transferTo(0, in.size(), out);
        in.close();
        out.close();
    }
Flag: $flag_false
--
"""
)

example_one_ojclone = Template(
    """
## Example
Code snippet 1: 
int main (){
	int n,i,m,a=0,t;
	scanf("%d",&n);
	for (i=1;i<=n*n;i++)

	{scanf("%d",&m);
	if (m==0)
		a=a+1;
	}
	
	t=(a+4)/4;
	printf("%d",(t-2)*(t-2));
	return 0;
}
Code snippet 2:
int main ( )
int main(int argc, char* argv[])
{
	int n,i=0,j;
	int N,k,p;
	scanf ("%d",&n);
	N=n*n;
	for (k=0;k<N;k++)
	{
		scanf ("%d",&p);
		if (p==0)
			i++;
	}
	j=((i/4)-1)*((i/4)-1);
    printf ("%d\n",j);
}
Flag: $flag_true
--
"""
)

example_few_ojclone = Template(
    """
## Example
Code snippet 1: 
int main (){
	int n,i,m,a=0,t;
	scanf("%d",&n);
	for (i=1;i<=n*n;i++)

	{scanf("%d",&m);
	if (m==0)
		a=a+1;
	}
	
	t=(a+4)/4;
	printf("%d",(t-2)*(t-2));
	return 0;
}
Code snippet 2:
int main ( )
int main(int argc, char* argv[])
{
	int n,i=0,j;
	int N,k,p;
	scanf ("%d",&n);
	N=n*n;
	for (k=0;k<N;k++)
	{
		scanf ("%d",&p);
		if (p==0)
			i++;
	}
	j=((i/4)-1)*((i/4)-1);
    printf ("%d\n",j);
}
Flag: $flag_true
--
## Example
Code snippet 1:
int main()
{
	int a;
         scanf ("%d",&a);
	if (a==9)
	    printf ("9\n");
	else if (a==6)
		printf ("2\n");
	return 0;
}
Code snippet 2:     
int main() {
	int n;
	cin >> n;
	int a[n];
	for(int i1=0;i1<n;i1++)
		cin >> a[i1];
	int j=0;
	for(int i1=0;i1<n;i1++){
		j=i1+1;
		for (int i2=i1+1;i2<n;i2++)
		{
			if(a[i2]!=a[i1])
			{a[j]=a[i2];j++;}
		}
		n=j;
	}
	for(int i1=0;i1<n;i1++){
	cout << a[i1];
	if (i1!=n-1)
		cout << " ";
	}
	return 0;
}

Flag: $flag_false
--
## Example
Code snippet 1: 
int main(){
	int row,col;cin>>row>>col;
	int i;int a[10010],*p=a;
	for(i=0;i<row*col;i++){cin>>*p;p++;}
	p=a;
    for(int c=0;c<row+col;c++){
    	for(i=c;i>=0;i--){
    		if(i<col&&c-i<row){cout<<*(p+i+(c-i)*col)<<endl;}
    	}
    }
	return 0;
}
Code snippet 2:
int main()
{
	int n=0,x=0,a[100]={0},i=0;
	cin>>n;
	cin>>x;
	cout<<x;
	a[x-1]=1;
	for(i=2;i<=n;i++)
	{
		cin>>x;
		if(a[x-1]==0)
		{
			cout<<" "<<x;
			a[x-1]=1;
		}
	}
	return 0;
}
Flag: $flag_false
--
## Example
Code snippet 1: 
int main()
{
	int n,k,a[1000];
	cin>>n>>k;
	for (int i=0;i<n;i++) cin>>a[i];
	for (int i=0;i<n;i++){
		for (int j=0;j<n;j++) {
			if (a[j]==k-a[i]) {
				cout<<"yes";
				return 0;
			}
		}
	} 
cout<<"no";
return 0;
}
Code snippet 2:
int main()
{
	int n,k,i,j,flag;
	int a[1000];
	cin>>n>>k;
	flag=0;
	for (i=0;i<n;i++)
	{
		cin>>a[i];
		for(j=0;j<i;j++)
			if (a[j]+a[i]==k) flag=1;
	}
	if (flag==1) cout<<"yes"<<endl;
	else cout<<"no"<<endl;
	return 0;
}
Flag: $flag_true
__
"""
)

input = """
Code snippet 1: $code_1
Code snippet 2: $code_2
"""

output = """
$output
"""

class PromptTemplate:
    def __init__(self, true_flag, false_flag):
        self.task = task
        self.instruction_one = instruction_one.substitute({"flag_true": true_flag, "flag_false": false_flag})
        self.instruction_few = instruction_few.substitute({"flag_true": true_flag, "flag_false": false_flag})
        self.definition = definition
        self.output_requirement = output_requirement.substitute({"flag_true": true_flag, "flag_false": false_flag})
        self.example_few_bcb = example_few_bcb.substitute({"flag_true": true_flag, "flag_false": false_flag})
        self.example_one_bcb = example_one_bcb.substitute({"flag_true": true_flag, "flag_false": false_flag})
        self.example_one_ojclone = example_one_ojclone.substitute({"flag_true": true_flag, "flag_false": false_flag})
        self.example_few_ojclone = example_few_ojclone.substitute({"flag_true": true_flag, "flag_false": false_flag})
        self.input = input
        self.prefix = """
<s>
[INST]
<<SYS>>
"""
        self.midfix = "<</SYS>>"
        self.suffix = """
[/INST]
</s>
"""
        self.output = output


class PromptTemplateBCB(PromptTemplate):
    def __init__(self, true_flag, false_flag):
        super().__init__(true_flag, false_flag)

    def generate_one_shot(self):
        temp = Template(
            "(task) " + self.task + "\n" +
            "(definition) " + self.definition + "\n" +
            "(Instructions) " + self.instruction_one + "\n" +
            "(output_requirement )" + self.output_requirement + "\n" +
            "(example) " + self.example_one_bcb + "\n" +
            "(input) " + self.input + "\n"
        )
        return temp

    def generate_few_shot(self):
        temp = Template(
            "(task) " + self.task + "\n" +
            "(definition) " + self.definition + "\n" +
            "(Instructions) " + self.instruction_few + "\n" +
            "(output_requirement )" + self.output_requirement + "\n" +
            "(example) " + self.example_few_bcb + "\n" +
            "(input) " + self.input + "\n"
        )
        return temp

    def generate_zero_shot(self):
        temp = Template(
            "(task) " + self.task + "\n" +
            "(definition) " + self.definition + "\n" +
            "(output_requirement )" + self.output_requirement + "\n" +
            "(input) " + self.input + "\n"
        )
        return temp


class PromptTemplateOJclone(PromptTemplate):
    def __init__(self, true_flag, false_flag):
        super().__init__(true_flag, false_flag)

    def generate_one_shot(self):
        temp = Template(
            "(task) " + self.task + "\n" +
            "(definition) " + self.definition + "\n" +
            "(Instructions) " + self.instruction_one + "\n" +
            "(output_requirement )" + self.output_requirement + "\n" +
            "(example) " + self.example_one_ojclone + "\n" +
            "(input) " + self.input + "\n"
        )
        return temp

    def generate_few_shot(self):
        temp = Template(
            "(task) " + self.task + "\n" +
            "(definition) " + self.definition + "\n" +
            "(Instructions) " + self.instruction_few + "\n" +
            "(output_requirement )" + self.output_requirement + "\n" +
            "(example) " + self.example_few_ojclone + "\n" +
            "(input) " + self.input + "\n"
        )

        return temp

    def generate_zero_shot(self):
        temp = Template(
            "(task) " + self.task + "\n" +
            "(definition) " + self.definition + "\n" +
            "(output_requirement )" + self.output_requirement + "\n" +
            "(input) " + self.input + "\n"
        )
        return temp


__REGISTERED_PROMPT_TEMPLATES = {"Reid996/big_clone_bench": PromptTemplateBCB,
                                 "Reid996/OJClone_code_clone_unbalanced": PromptTemplateOJclone}


def build_prompt(cfg):
    flag_true = cfg.PROMPT.INDICATE_TRUE
    flag_false = cfg.PROMPT.INDICATE_FALSE

    prompt_group = __REGISTERED_PROMPT_TEMPLATES[cfg.DATA.NAME](flag_true, flag_false)

    if cfg.PROMPT.NAME == 'zero_shot':
        return prompt_group.generate_zero_shot()
    elif cfg.PROMPT.NAME == 'one_shot':
        return prompt_group.generate_one_shot()
    elif cfg.PROMPT.NAME == 'few_shot':
        return prompt_group.generate_few_shot()
    else:
        print("Unknown prompt template")

    return None


if __name__ == '__main__':
    xx = Template("dsad $2")

    xx1 = Template("dsad $1")
