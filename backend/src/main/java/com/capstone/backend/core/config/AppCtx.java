package com.capstone.backend.core.config;


import org.apache.tomcat.jdbc.pool.DataSource;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.EnableAspectJAutoProxy;
import org.springframework.jdbc.datasource.DataSourceTransactionManager;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.web.reactive.function.client.WebClient;


@Configuration 
@EnableAspectJAutoProxy
public class AppCtx{ 

	@Value("${spring.datasource.driver-class-name}")
	private String driverClassName;

	@Value("${spring.datasource.url}")
	private String url;

	@Value("${spring.datasource.username}")
	private String username;

	@Value("${spring.datasource.password}")
	private String password;


	// dataSource 빈 등록 
	@Bean(destroyMethod = "close")
	public DataSource dataSource() {
		
		DataSource ds = new DataSource();
		ds.setDriverClassName(driverClassName);
		ds.setUrl(url);
		ds.setUsername(username);
		ds.setPassword(password);

		ds.setInitialSize(10);
		ds.setMaxActive(10000);

		return ds; 
	}

	// 트랜잭션 매니저 빈 등록 
	@Bean
	public PlatformTransactionManager transactionManager() {
		DataSourceTransactionManager tm = new DataSourceTransactionManager(); 
		tm.setDataSource(dataSource());
		return tm ;
	}


//	@Bean
//	public WebClient webClient(){
//		return WebClient.builder()
//				.baseUrl("https://www.alphavantage.co")
//				.build();
//	}

	@Bean
	public WebClient webClient(WebClient.Builder builder){
		return builder.build();
	}

}







